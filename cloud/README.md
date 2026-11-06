# Cloud run (Lambda A100)

Token: `LAMBDA_API_KEY` from the environment only (source .env first; never print it).
Sequence (from project root, quote the path; IP/ID are placeholders):

1. `python cloud/lambda_client.py types --filter a100` (confirm gpu_1x_a100_sxm4 in us-east-1, $1.99/h)
2. `ID=$(python cloud/lambda_client.py launch --type gpu_1x_a100_sxm4 --region us-east-1 --ssh-key lambda-mnist --name mnist-ddpm-rf)`
3. `IP=$(python cloud/lambda_client.py wait --id $ID)`
4. `bash cloud/sync.sh $IP push`
5. `ssh -i ~/.ssh/lambda-mnist ubuntu@$IP 'bash ~/mnist-flow/cloud/remote_setup.sh'`
6. `ssh -i ~/.ssh/lambda-mnist ubuntu@$IP 'cd ~/mnist-flow && mkdir -p runs && nohup bash cloud/remote_run.sh > runs/remote_run.out 2>&1 &'`
7. Loop: `bash cloud/sync.sh $IP status` until `all done` or `FAILED <step>`;
   `python cloud/lambda_client.py cost --id $ID` each check.
8. `bash cloud/sync.sh $IP pull` and verify checkpoints/*/final.pt, results/metrics.json exist.
9. `python cloud/lambda_client.py terminate --id $ID --yes`, then `instances` until empty.

Cost math: $1.99/h x hours (ledger: .lambda-run/mnist-ddpm-rf.{md,json}).
Budget $18; hard stop at $10 projected (about 5 h).

Rules: never `rm`/`rm -rf` (use `mv`); always pull before terminating; terminate only
with explicit approval; the ledger is the record of what was billed; Lambda sits behind
Cloudflare so a browser User-Agent is sent on every request.
