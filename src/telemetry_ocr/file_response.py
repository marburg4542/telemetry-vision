"""A FileResponse that can send a file bigger than Windows sends in one call.

asyncio sends a file on Windows with TransmitFile, which refuses more than
2 GiB - 2 bytes at a time: past that it fails with WinError 87 before a byte is
written, and the page gets an empty response. A 45-minute flight is 2.4-2.7 GB,
and the first request a <video> makes for it is "bytes=0-" -- the whole file --
so those flights would not play at all. The body is sent in pieces instead,
each well under the limit, one after another on the same connection.

Imported only where aiohttp is: the web extra is optional.
"""

from __future__ import annotations

import asyncio
from typing import IO, Any

from aiohttp import web
from aiohttp.abc import AbstractStreamWriter
from aiohttp.web_request import BaseRequest

# Half of the limit: a round number with room to spare.
SENDFILE_PIECE = 1 << 30


class LargeFileResponse(web.FileResponse):
    async def _sendfile(
        self, request: BaseRequest, fobj: IO[Any], offset: int, count: int
    ) -> AbstractStreamWriter:
        if count <= SENDFILE_PIECE:
            return await super()._sendfile(request, fobj, offset, count)
        writer = await web.StreamResponse.prepare(self, request)
        assert writer is not None
        loop = asyncio.get_running_loop()
        while count > 0:
            transport = request.transport
            if transport is None:
                raise ConnectionResetError("Connection lost")
            piece = min(count, SENDFILE_PIECE)
            try:
                await loop.sendfile(transport, fobj, offset, piece)
            except NotImplementedError:
                return await self._sendfile_fallback(writer, fobj, offset, count)
            offset += piece
            count -= piece
        await web.StreamResponse.write_eof(self)
        return writer
