import argparse
import asyncio
import datetime
import json
import os
import sys
import urllib.parse

import dotenv
from polar_token_manager import PolarTokenManager

#
# To bootstrap auth:
#
# * Run this script with 'auth-url'
# * Open the returned URL in a browser and authenticate in the Polar webpage
# * Extract the 'code' from the redirect URL that you are sent to (which gives 404)
# * Run with --code <code> from the previous step and cross fingers
# If successful, you will get json with tokens


def build_authorization_url(
    client_id: str,
    redirect_uri: str,
    scope: str,
    state: str,
) -> str:
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scope,
        "state": state,
    }
    return "https://auth.polar.com/oauth/authorize?" + urllib.parse.urlencode(params)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Async Polar AccessLink CLI")
    parser.add_argument(
        "--client-id",
        default=os.environ.get("POLAR_V4_CLIENT_ID"),
        help="Polar OAuth client id or set POLAR_CLIENT_ID",
    )
    parser.add_argument(
        "--client-secret",
        default=os.environ.get("POLAR_V4_CLIENT_SECRET"),
        help="Polar OAuth client secret or set POLAR_CLIENT_SECRET",
    )
    parser.add_argument(
        "--token-file",
        default=os.environ.get("POLAR_TOKEN_FILE", "/home/berland/.polar_tokens.json"),
        help="Path to token JSON file",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    auth_parser = subparsers.add_parser(
        "auth-url", help="Print Polar authorization URL"
    )
    auth_parser.add_argument(
        "--redirect-uri",
        required=True,
        help="Exact redirect URI registered in Polar",
    )
    auth_parser.add_argument(
        "--scope",
        required=True,
        help="Space-separated scopes to request",
    )
    auth_parser.add_argument(
        "--state",
        default="polar-cli",
        help="OAuth state parameter",
    )

    init_parser = subparsers.add_parser("init", help="Exchange authorization code")
    init_parser.add_argument(
        "--code",
        required=True,
        help="Authorization code returned by Polar OAuth redirect",
    )

    subparsers.add_parser("token", help="Print a valid access token")
    subparsers.add_parser("hello", help="Call Polar v4 hello endpoint")

    nr_parser = subparsers.add_parser(
        "nightly-recharge", help="Fetch v4 Nightly Recharge data"
    )
    nr_parser.add_argument(
        "--from-date",
        default=(datetime.datetime.now() - datetime.timedelta(days=1))
        .date()
        .isoformat(),
        help="Start date YYYY-MM-DD",
    )
    nr_parser.add_argument(
        "--to-date",
        default=datetime.datetime.now().date().isoformat(),
        help="End date YYYY-MM-DD",
    )

    return parser


async def async_main() -> int:  # noqa: PLR0911
    dotenv.load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    if not args.client_id:
        print("ERROR: missing client id", file=sys.stderr)
        return 2

    if args.command == "auth-url":
        url = build_authorization_url(
            client_id=args.client_id,
            redirect_uri=args.redirect_uri,
            scope=args.scope,
            state=args.state,
        )
        print(url)
        return 0

    if not args.client_secret:
        print(
            "ERROR: provide --client-secret or set POLAR_CLIENT_SECRET",
            file=sys.stderr,
        )
        return 2

    manager = PolarTokenManager(
        client_id=args.client_id,
        client_secret=args.client_secret,
        token_file=args.token_file,
    )

    if args.command == "init":
        token_data = await manager.exchange_authorization_code(args.code)
        print(json.dumps(token_data, indent=2, sort_keys=True))
        return 0

    if args.command == "token":
        access_token = await manager.get_valid_access_token()
        print(access_token)
        return 0

    if args.command == "hello":
        result = await manager.authorized_get(
            "https://www.polaraccesslink.com/v4/data/hello",
            accept="text/plain",
        )
        print(result.get("text", result))
        return 0

    if args.command == "nightly-recharge":
        result = await manager.authorized_get(
            "https://www.polaraccesslink.com/v4/data/nightly-recharge-results",
            params={
                "from": args.from_date,
                "to": args.to_date,
                "features": "",  # "samples"
            },
            accept="application/json",
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    print(f"ERROR: unknown command {args.command}", file=sys.stderr)
    return 2


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
