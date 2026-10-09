"""离线闸自检：conftest 的 _block_external_network 必须真的能拦。

这类「守卫的守卫」专防一种事故：fixture 写错（比如条件恒假）后，
离线约束悄悄失效、测试却依旧全绿——与本项目「假绿」教训同源。
"""
from __future__ import annotations

import socket

import pytest


def test_非回环连接必须被拦() -> None:
    with pytest.raises(RuntimeError, match="离线约束"):
        socket.socket().connect(("8.8.8.8", 53))


def test_非回环DNS解析必须被拦() -> None:
    with pytest.raises(RuntimeError, match="离线约束"):
        socket.getaddrinfo("example.com", 443)


def test_回环必须放行否则异步用例假失败() -> None:
    # 不真连：只要求 getaddrinfo(localhost) 不触发拦截
    socket.getaddrinfo("localhost", 9)
    socket.getaddrinfo("127.0.0.1", 9)
