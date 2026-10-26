#!/usr/bin/env bash
# Detached NIAH: owns the server lifecycle so the grid never races a half-loaded model.
set -u
cd "$HOME/glm-5.3-flash-cuda-server"
mkdir -p logs artifacts
CTX=${CTX:-132096}
./build/glm5-server --ctx "$CTX" --port 8080 > logs/niah_server.log 2>&1 &
SRV=$!
echo "server pid $SRV, ctx $CTX"
# readiness: poll the endpoint, not the log (the log raced us last time)
for i in $(seq 1 3000); do
  if curl -sf -m 5 http://127.0.0.1:8080/health >/dev/null 2>&1; then echo "ready after ${i}0s"; break; fi
  if ! kill -0 "$SRV" 2>/dev/null; then echo "SERVER DIED"; tail -20 logs/niah_server.log; exit 1; fi
  sleep 10
done
curl -sf -m 10 http://127.0.0.1:8080/props | head -c 400; echo
python3 tools/niah_nvfp4.py --lengths "${LENGTHS:-32000,64000,128000}" \
        --depths "${DEPTHS:-0.1,0.5,0.9}" --out artifacts/niah_nvfp4_dsa.json
rc=$?
echo "niah rc=$rc"
kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
echo "DONE rc=$rc"
