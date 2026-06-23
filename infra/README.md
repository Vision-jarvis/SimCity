# SimCity Infrastructure

Zero-cost deployment to the **Oracle Cloud Always-Free** tier (4 OCPU / 24 GB
Ampere A1) running **k3s** (lightweight Kubernetes). Everything here stays
within free allowances.

```
infra/
├── terraform/   # Provision the OCI VM + network + k3s bootstrap
├── k8s/         # Kubernetes manifests (kustomize)
└── helm/        # Helm values surface for chart-based installs
```

## 1. Provision the node (Terraform)

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars   # fill in OCI creds
terraform init
terraform apply
terraform output service_urls
```

The instance bootstraps k3s via cloud-init; the kubeconfig lands at
`/home/ubuntu/.kube/config`.

## 2. Build & push images

```bash
docker build -t ghcr.io/your-org/simcity-api:latest .
docker build -t ghcr.io/your-org/simcity-frontend:latest ./frontend
docker push ghcr.io/your-org/simcity-api:latest
docker push ghcr.io/your-org/simcity-frontend:latest
```

(The CI `deploy.yml` workflow can do this automatically on merge to `main`.)

## 3. Deploy the stack

```bash
# Point kubectl at the k3s node, then:
kubectl apply -k infra/k8s

# Set your real image tags without editing files:
cd infra/k8s
kustomize edit set image simcity-api=ghcr.io/your-org/simcity-api:<sha>
kustomize edit set image simcity-frontend=ghcr.io/your-org/simcity-frontend:<sha>
```

Frontend is exposed on NodePort `30000`; API/Neo4j/Grafana ports are opened by
the Terraform security list.

## Helm alternative

`helm/values.yaml` is the single tunable surface if you prefer a Helm umbrella
chart over kustomize. Override per-environment with `-f values.prod.yaml`.
