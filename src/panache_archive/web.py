"""HTTP reads with retries: the ADS and Open-Meteo answer a 502 now and then."""

import logging
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Final

ATTEMPTS: Final = 4
FIRST_WAIT_SECONDS: Final = 2.0
"""Waits double after each failure: 2, 4 then 8 seconds."""
TIMEOUT_SECONDS: Final = 60

logger = logging.getLogger(__name__)


def read(
    request: urllib.request.Request | str, *, sleep: Callable[[float], None] = time.sleep
) -> bytes:
    """The answer's body. Server errors (5xx), lost connections and timeouts are tried again;
    other errors, such as a 400 or a 429, are raised at once."""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as answer:
                return bytes(answer.read())
        except urllib.error.HTTPError as error:
            if error.code < 500 or attempt == ATTEMPTS:
                raise
            reason = f"HTTP {error.code}"
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == ATTEMPTS:
                raise
            reason = str(error)
        wait = FIRST_WAIT_SECONDS * 2 ** (attempt - 1)
        logger.warning("%s on attempt %d of %d, again in %.0f s", reason, attempt, ATTEMPTS, wait)
        sleep(wait)
    raise AssertionError("unreachable: the last attempt returns or raises")
