"""Blocking facades backed by one privately owned asynchronous event loop."""

from __future__ import annotations

import asyncio
import importlib
import threading
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import Future
from types import TracebackType
from typing import Protocol, TypeVar, cast

from penampakan.config import Settings
from penampakan.errors import SessionClosedError, SyncInAsyncContextError
from penampakan.models import (
    ImageAsset,
    ImageSource,
    InspectionPlan,
    InspectionResult,
    Observation,
    VisionAnswer,
)
from penampakan.protocols import ActionPolicy, Cache, TextLLM, TraceSink, VisionBackend

_T = TypeVar("_T")


class _AsyncSession(Protocol):
    @property
    def root_asset(self) -> ImageAsset: ...

    @property
    def assets(self) -> tuple[ImageAsset, ...]: ...

    @property
    def observations(self) -> tuple[Observation, ...]: ...

    async def inspect(
        self,
        plan: InspectionPlan | None = None,
        *,
        timeout_s: float | None = None,
    ) -> InspectionResult: ...

    async def ask(
        self,
        question: str,
        *,
        timeout_s: float | None = None,
    ) -> VisionAnswer: ...

    def get_asset(self, asset_id: str) -> ImageAsset: ...

    def get_observation(self, observation_id: str) -> Observation: ...

    async def aclose(self) -> None: ...

    async def __aenter__(self) -> _AsyncSession: ...


class _AsyncClient(Protocol):
    @property
    def settings(self) -> Settings: ...

    @property
    def closed(self) -> bool: ...

    async def open_image(self, source: ImageSource) -> _AsyncSession: ...

    async def inspect(
        self,
        source: ImageSource,
        plan: InspectionPlan | None = None,
        *,
        timeout_s: float | None = None,
    ) -> InspectionResult: ...

    async def ask(
        self,
        source: ImageSource,
        question: str,
        *,
        timeout_s: float | None = None,
    ) -> VisionAnswer: ...

    async def aclose(self) -> None: ...

    async def __aenter__(self) -> _AsyncClient: ...


def _reject_running_loop() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise SyncInAsyncContextError()


async def _invoke(factory: Callable[[], Awaitable[_T]]) -> _T:
    return await factory()


async def _read(getter: Callable[[], _T]) -> _T:
    return getter()


class _LoopThread:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._finished = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._startup_error: BaseException | None = None
        self._stopped = False

    def run(self, factory: Callable[[], Awaitable[_T]]) -> _T:
        loop = self._start()
        invocation = _invoke(factory)
        try:
            future: Future[_T] = asyncio.run_coroutine_threadsafe(invocation, loop)
        except BaseException:
            invocation.close()
            raise
        try:
            return future.result()
        except BaseException:
            future.cancel()
            raise

    def _start(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._stopped:
                raise SessionClosedError()
            if self._thread is None:
                thread = threading.Thread(
                    target=self._serve,
                    name="penampakan-sync-loop",
                    daemon=True,
                )
                self._thread = thread
                try:
                    thread.start()
                except BaseException as error:
                    self._startup_error = error
                    self._stopped = True
                    self._ready.set()
                    self._finished.set()
                    raise
            ready = self._ready
        ready.wait()
        with self._lock:
            if self._startup_error is not None:
                raise self._startup_error
            if self._loop is None:
                raise RuntimeError("background event loop did not start")
            return self._loop

    def _serve(self) -> None:
        loop: asyncio.AbstractEventLoop | None = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            with self._lock:
                self._loop = loop
            self._ready.set()
            loop.run_forever()
        except BaseException as error:
            with self._lock:
                self._startup_error = error
            self._ready.set()
        finally:
            try:
                if loop is not None:
                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        task.cancel()
                    if pending:
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    loop.run_until_complete(loop.shutdown_asyncgens())
                    loop.run_until_complete(loop.shutdown_default_executor())
                    asyncio.set_event_loop(None)
                    loop.close()
            finally:
                self._finished.set()

    def stop(self) -> None:
        with self._lock:
            if self._stopped:
                thread = self._thread
            else:
                self._stopped = True
                thread = self._thread
                loop = self._loop
                if loop is not None and loop.is_running():
                    loop.call_soon_threadsafe(loop.stop)
        if thread is None:
            return
        self._finished.wait()
        if thread is not threading.current_thread():
            thread.join()


class _ClientBridge:
    def __init__(self, client: _AsyncClient) -> None:
        self._client = client
        self._runner = _LoopThread()
        self._condition = threading.Condition()
        self._state = "open"
        self._active_calls = 0
        self._close_error: BaseException | None = None

    def run(self, factory: Callable[[], Awaitable[_T]]) -> _T:
        _reject_running_loop()
        with self._condition:
            if self._state != "open":
                raise SessionClosedError()
            self._active_calls += 1
        try:
            return self._runner.run(factory)
        finally:
            with self._condition:
                self._active_calls -= 1
                self._condition.notify_all()

    def close(self) -> None:
        _reject_running_loop()
        with self._condition:
            if self._state == "closed":
                self._raise_close_error()
                return
            if self._state == "closing":
                while self._state != "closed":
                    self._condition.wait()
                self._raise_close_error()
                return
            self._state = "closing"
            try:
                while self._active_calls:
                    self._condition.wait()
            except BaseException:
                self._state = "open"
                self._condition.notify_all()
                raise
        error: BaseException | None = None
        try:
            self._runner.run(self._client.aclose)
        except BaseException as close_error:
            error = close_error
        try:
            self._runner.stop()
        except BaseException as stop_error:
            if error is None:
                error = stop_error
        with self._condition:
            self._close_error = error
            self._state = "closed"
            self._condition.notify_all()
        if error is not None:
            raise error

    def wait_for_close(self) -> bool:
        with self._condition:
            if self._state == "open":
                return False
            while self._state != "closed":
                self._condition.wait()
            self._raise_close_error()
            return True

    def _raise_close_error(self) -> None:
        if self._close_error is not None:
            raise self._close_error


class VisionSession:
    """Blocking view of one asynchronous reusable vision session."""

    def __init__(self, session: _AsyncSession, bridge: _ClientBridge) -> None:
        self._session = session
        self._bridge = bridge
        self._condition = threading.Condition()
        self._state = "open"
        self._active_calls = 0
        self._close_error: BaseException | None = None

    @property
    def root_asset(self) -> ImageAsset:
        """Return the immutable root image metadata."""
        return self._run(lambda: _read(lambda: self._session.root_asset))

    @property
    def assets(self) -> tuple[ImageAsset, ...]:
        """Return an immutable snapshot of all current image assets."""
        return self._run(lambda: _read(lambda: self._session.assets))

    @property
    def observations(self) -> tuple[Observation, ...]:
        """Return an immutable snapshot of all current observations."""
        return self._run(lambda: _read(lambda: self._session.observations))

    def inspect(
        self,
        plan: InspectionPlan | None = None,
        *,
        timeout_s: float | None = None,
    ) -> InspectionResult:
        """Run a blocking inspection on this session."""
        return self._run(lambda: self._session.inspect(plan, timeout_s=timeout_s))

    def ask(
        self,
        question: str,
        *,
        timeout_s: float | None = None,
    ) -> VisionAnswer:
        """Answer one visual question while retaining session observations."""
        return self._run(lambda: self._session.ask(question, timeout_s=timeout_s))

    def get_asset(self, asset_id: str) -> ImageAsset:
        """Look up one asset by its stable identifier."""
        return self._run(lambda: _read(lambda: self._session.get_asset(asset_id)))

    def get_observation(self, observation_id: str) -> Observation:
        """Look up one observation by its stable identifier."""
        return self._run(lambda: _read(lambda: self._session.get_observation(observation_id)))

    def close(self) -> None:
        """Close the asynchronous session exactly once."""
        _reject_running_loop()
        with self._condition:
            if self._state == "closed":
                self._raise_close_error()
                return
            if self._state == "closing":
                while self._state != "closed":
                    self._condition.wait()
                self._raise_close_error()
                return
            self._state = "closing"
            try:
                while self._active_calls:
                    self._condition.wait()
            except BaseException:
                self._state = "open"
                self._condition.notify_all()
                raise
        error: BaseException | None = None
        try:
            self._bridge.run(self._session.aclose)
        except SessionClosedError as close_error:
            try:
                if not self._bridge.wait_for_close():
                    error = close_error
            except BaseException as bridge_error:
                error = bridge_error
        except BaseException as close_error:
            error = close_error
        with self._condition:
            self._close_error = error
            self._state = "closed"
            self._condition.notify_all()
        if error is not None:
            raise error

    def _run(self, factory: Callable[[], Awaitable[_T]]) -> _T:
        _reject_running_loop()
        with self._condition:
            if self._state != "open":
                raise SessionClosedError()
            self._active_calls += 1
        try:
            return self._bridge.run(factory)
        finally:
            with self._condition:
                self._active_calls -= 1
                self._condition.notify_all()

    def _raise_close_error(self) -> None:
        if self._close_error is not None:
            raise self._close_error

    def __enter__(self) -> VisionSession:
        """Return this open blocking session."""
        self._run(self._session.__aenter__)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the session when leaving a synchronous context."""
        self.close()


class Penampakan:
    """Blocking client backed by one lazily created event-loop thread."""

    def __init__(
        self,
        *,
        llm: TextLLM | None = None,
        backends: Sequence[VisionBackend] = (),
        policy: ActionPolicy | None = None,
        cache: Cache | None = None,
        settings: Settings | None = None,
        trace_sinks: Sequence[TraceSink] = (),
        owns_policy: bool = False,
        owns_llm: bool = False,
    ) -> None:
        module = importlib.import_module("penampakan.client")
        factory = cast(Callable[..., _AsyncClient], module.AsyncPenampakan)
        client = factory(
            llm=llm,
            backends=backends,
            policy=policy,
            cache=cache,
            settings=settings,
            trace_sinks=trace_sinks,
            owns_policy=owns_policy,
            owns_llm=owns_llm,
        )
        self._client = client
        self._bridge = _ClientBridge(client)

    @property
    def settings(self) -> Settings:
        """Return the immutable client settings."""
        return self._client.settings

    @property
    def closed(self) -> bool:
        """Return whether the asynchronous client has closed."""
        return self._client.closed

    def open_image(self, source: ImageSource) -> VisionSession:
        """Open one reusable blocking image session."""
        session = self._bridge.run(lambda: self._client.open_image(source))
        return VisionSession(session, self._bridge)

    def inspect(
        self,
        source: ImageSource,
        plan: InspectionPlan | None = None,
        *,
        timeout_s: float | None = None,
    ) -> InspectionResult:
        """Run a blocking one-shot image inspection."""
        return self._bridge.run(lambda: self._client.inspect(source, plan, timeout_s=timeout_s))

    def ask(
        self,
        source: ImageSource,
        question: str,
        *,
        timeout_s: float | None = None,
    ) -> VisionAnswer:
        """Run a blocking one-shot visual question."""
        return self._bridge.run(lambda: self._client.ask(source, question, timeout_s=timeout_s))

    def close(self) -> None:
        """Close the asynchronous client and join its private loop thread."""
        self._bridge.close()

    def __enter__(self) -> Penampakan:
        """Return this blocking client."""
        self._bridge.run(self._client.__aenter__)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the client when leaving a synchronous context."""
        self.close()


__all__ = ["Penampakan", "VisionSession"]
