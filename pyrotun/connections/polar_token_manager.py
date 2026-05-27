import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp

TOKEN_URL = "https://auth.polar.com/oauth/token"


class PolarTokenManager:
    """This is for v4 of the Polar API, it is independent of the existing accesslink
    code in thie repo."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        token_file: str = "polar_tokens.json",
        refresh_skew_seconds: int = 300,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_file = Path(token_file)
        self.refresh_skew_seconds = refresh_skew_seconds
        self._lock = asyncio.Lock()

    async def load_tokens(self) -> Dict[str, Any]:
        if not self.token_file.exists():
            return {}
        return json.loads(self.token_file.read_text())

    async def save_tokens(self, token_data: Dict[str, Any]) -> None:
        token_data = dict(token_data)
        expires_in = token_data.get("expires_in")
        if expires_in is not None:
            token_data["expires_at"] = int(time.time()) + int(expires_in)
        self.token_file.write_text(json.dumps(token_data, indent=2, sort_keys=True))

    async def _post_token(self, data: Dict[str, Any]) -> Dict[str, Any]:
        auth = aiohttp.BasicAuth(self.client_id, self.client_secret, encoding="utf-8")
        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(auth=auth, timeout=timeout) as session:
            async with session.post(TOKEN_URL, data=data) as response:
                text = await response.text()
                if response.status >= 400:
                    raise RuntimeError(
                        f"Token request failed: HTTP {response.status}: {text}"
                    )
                return await response.json()

    async def exchange_authorization_code(
        self,
        authorization_code: str,
        redirect_uri: str = "http://localhost:5000/oauth2_callback",
    ) -> Dict[str, Any]:
        body = {
            "grant_type": "authorization_code",
            "code": authorization_code,
        }
        if redirect_uri:
            body["redirect_uri"] = redirect_uri

        token_data = await self._post_token(body)
        await self.save_tokens(token_data)
        return token_data

    async def refresh_access_token(self) -> Dict[str, Any]:
        tokens = await self.load_tokens()
        refresh_token = tokens.get("refresh_token")
        if not refresh_token:
            raise RuntimeError("No refresh_token found in token storage.")

        token_data = await self._post_token(
            {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            }
        )

        if "refresh_token" not in token_data:
            token_data["refresh_token"] = refresh_token

        await self.save_tokens(token_data)
        return token_data

    def token_is_expiring(self, tokens: Dict[str, Any]) -> bool:
        expires_at = tokens.get("expires_at")
        if not expires_at:
            return True
        return time.time() >= (int(expires_at) - self.refresh_skew_seconds)

    async def get_valid_access_token(self) -> str:
        async with self._lock:
            tokens = await self.load_tokens()
            if not tokens:
                raise RuntimeError(
                    "No tokens found. Run exchange_authorization_code() first."
                )

            if self.token_is_expiring(tokens):
                tokens = await self.refresh_access_token()

            access_token = tokens.get("access_token")
            if not access_token:
                raise RuntimeError("No access_token available after refresh/load.")
            return access_token

    async def authorized_get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        accept: str = "application/json",
    ) -> Dict[str, Any]:
        access_token = await self.get_valid_access_token()
        timeout = aiohttp.ClientTimeout(total=60)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                url,
                params=params,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": accept,
                },
            ) as response:
                if response.status == 401:
                    async with self._lock:
                        refreshed = await self.refresh_access_token()
                        access_token = refreshed["access_token"]

                    async with session.get(
                        url,
                        params=params,
                        headers={
                            "Authorization": f"Bearer {access_token}",
                            "Accept": accept,
                        },
                    ) as retry_response:
                        text = await retry_response.text()
                        if retry_response.status >= 400:
                            raise RuntimeError(
                                f"GET failed after refresh: HTTP {retry_response.status}: {text}"
                            )
                        if accept == "application/json":
                            return await retry_response.json()
                        return {"text": text}

                text = await response.text()
                if response.status >= 400:
                    raise RuntimeError(f"GET failed: HTTP {response.status}: {text}")

                if accept == "application/json":
                    return await response.json()
                return {"text": text}
