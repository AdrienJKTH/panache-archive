#!/usr/bin/env bash
# Prints an access token of the Copernicus Land Monitoring Service's API (valid one hour), from
# the service key in $CLMS_SERVICE_KEY (the JSON a CLMS profile issues under "API Tokens"): a JWT
# signed RS256 with the key's private half, traded at its token endpoint.
# https://eea.github.io/clms-api-docs/authentication.html
#
# Usage: token=$(scripts/clms-token.sh)
set -euo pipefail

: "${CLMS_SERVICE_KEY:?the CLMS service key, as JSON}"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
printf '%s' "$CLMS_SERVICE_KEY" > "$work/key.json"
jq -r .private_key "$work/key.json" > "$work/private.pem"

b64url() { openssl base64 -A | tr '+/' '-_' | tr -d '='; }
now=$(date +%s)
header=$(printf '{"alg":"RS256","typ":"JWT"}' | b64url)
claims=$(jq -c --argjson iat "$now" --argjson exp $((now + 3600)) \
  '{iss: .client_id, sub: .user_id, aud: .token_uri, iat: $iat, exp: $exp}' "$work/key.json" | b64url)
signature=$(printf '%s.%s' "$header" "$claims" \
  | openssl dgst -sha256 -sign "$work/private.pem" -binary | b64url)

curl -fsS -X POST "$(jq -r .token_uri "$work/key.json")" \
  -H 'Accept: application/json' -H 'Content-Type: application/x-www-form-urlencoded' \
  --data-urlencode 'grant_type=urn:ietf:params:oauth:grant-type:jwt-bearer' \
  --data-urlencode "assertion=$header.$claims.$signature" \
  | jq -re .access_token
