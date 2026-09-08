"""Offline checks for the opt-in network probe; these never contact the web."""
import asyncio
from pathlib import Path
import runpy

import pytest

from v_core.mcp_tools import MCPTools, MCPToolExecutionError


check_network_tools = runpy.run_path(str(
    Path(__file__).resolve().parents[1] / 'scripts' / 'check_network_tools.py'
))['check_network_tools']


class BrowserFixture(MCPTools):
    def __init__(self, *, empty_search=False, false_read=False, invalid_succeeds=False):
        self.calls = []
        self.url = ''
        self.empty_search = empty_search
        self.false_read = false_read
        self.invalid_succeeds = invalid_succeeds

    async def browser_call(self, name, arguments):
        self.calls.append((name, arguments))
        if name == 'browser_navigate':
            self.url = arguments['url']
            if self.url.endswith('.invalid/') and not self.invalid_succeeds:
                raise MCPToolExecutionError('fixture: domain unavailable')
            return 'navigation finished'
        if 'duckduckgo.com' in self.url:
            if self.empty_search:
                return '- Page Title: Search unavailable'
            return '- link "IANA Example Domains":\n  - /url: https://www.iana.org/help/example-domains'
        if self.false_read:
            return '- Page URL: https://example.com/\n- Page Title: Example Domain\n- paragraph: Access denied'
        if 'iana.org' in self.url:
            return f'- Page URL: {self.url}\n- Page Title: IANA Example Domains\n- paragraph: Example domains for documentation'
        return f'- Page URL: {self.url}\n- Page Title: Example Domain\n- paragraph: This domain is for documentation examples'


@pytest.mark.asyncio
async def test_fixed_network_cases_verify_data_with_offline_browser():
    tools = BrowserFixture()
    report = await check_network_tools(tools)
    assert report['counts'] == {'passed': 5, 'failed': 0}
    assert {name for name, _ in tools.calls} == {'browser_navigate', 'browser_snapshot'}


@pytest.mark.asyncio
@pytest.mark.parametrize('options,failed', [
    ({'empty_search': True}, {'public_web_search', 'read_search_result'}),
    ({'false_read': True}, {'read_public_example', 'browser_snapshot_consistency', 'read_search_result'}),
    ({'invalid_succeeds': True}, {'unavailable_domain_error'}),
])
async def test_network_probe_rejects_unsupported_success(options, failed):
    report = await check_network_tools(BrowserFixture(**options))
    assert {row['case'] for row in report['results'] if row['status'] == 'failed'} == failed


@pytest.mark.asyncio
async def test_network_probe_cancellation_is_not_swallowed():
    tools = BrowserFixture()
    async def cancelled(*args):
        raise asyncio.CancelledError
    tools.browser_call = cancelled
    with pytest.raises(asyncio.CancelledError):
        await check_network_tools(tools)


@pytest.mark.asyncio
async def test_network_timeouts_fail_instead_of_claiming_completion():
    tools = BrowserFixture()
    async def stalled(*args):
        await asyncio.sleep(60)
    tools.browser_call = stalled
    report = await check_network_tools(tools, timeout=0.01)
    assert report['counts'] == {'passed': 0, 'failed': 5}
    assert sum(row.get('error_type') == 'TimeoutError' for row in report['results']) == 4
