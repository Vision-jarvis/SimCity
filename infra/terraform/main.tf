# SimCity — Oracle Cloud Always-Free infrastructure.
#
# Provisions a single Ampere A1 (4 OCPU / 24 GB) Always-Free VM, a VCN with a
# public subnet, and a security list opening the SimCity service ports. The
# instance bootstraps k3s (lightweight Kubernetes) via cloud-init so the
# manifests under infra/k8s/ can be applied immediately after `terraform apply`.
#
# Estimated monthly cost: $0 (entirely within the OCI Always-Free allowances).

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    oci = {
      source  = "oracle/oci"
      version = ">= 5.0.0"
    }
  }
}

provider "oci" {
  tenancy_ocid     = var.tenancy_ocid
  user_ocid        = var.user_ocid
  fingerprint      = var.fingerprint
  private_key_path = var.private_key_path
  region           = var.region
}

# --- Networking ---------------------------------------------------------------

resource "oci_core_vcn" "simcity" {
  compartment_id = var.compartment_ocid
  display_name   = "simcity-vcn"
  cidr_blocks    = ["10.0.0.0/16"]
  dns_label      = "simcity"
}

resource "oci_core_internet_gateway" "simcity" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.simcity.id
  display_name   = "simcity-igw"
}

resource "oci_core_route_table" "simcity" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.simcity.id
  display_name   = "simcity-rt"

  route_rules {
    destination       = "0.0.0.0/0"
    network_entity_id = oci_core_internet_gateway.simcity.id
  }
}

resource "oci_core_security_list" "simcity" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.simcity.id
  display_name   = "simcity-sl"

  egress_security_rules {
    destination = "0.0.0.0/0"
    protocol    = "all"
  }

  # SSH, API (8000), frontend (3000), Grafana (3001), Neo4j browser (7474).
  dynamic "ingress_security_rules" {
    for_each = [22, 80, 443, 3000, 3001, 7474, 8000]
    content {
      protocol = "6" # TCP
      source   = "0.0.0.0/0"
      tcp_options {
        min = ingress_security_rules.value
        max = ingress_security_rules.value
      }
    }
  }
}

resource "oci_core_subnet" "simcity" {
  compartment_id    = var.compartment_ocid
  vcn_id            = oci_core_vcn.simcity.id
  cidr_block        = "10.0.1.0/24"
  display_name      = "simcity-public-subnet"
  route_table_id    = oci_core_route_table.simcity.id
  security_list_ids = [oci_core_security_list.simcity.id]
  dns_label         = "public"
}

# --- Compute ------------------------------------------------------------------

data "oci_identity_availability_domains" "ads" {
  compartment_id = var.tenancy_ocid
}

# Canonical Ubuntu 22.04 image for the A1 (aarch64) shape.
data "oci_core_images" "ubuntu" {
  compartment_id           = var.compartment_ocid
  operating_system         = "Canonical Ubuntu"
  operating_system_version = "22.04"
  shape                    = var.instance_shape
  sort_by                  = "TIMECREATED"
  sort_order               = "DESC"
}

resource "oci_core_instance" "simcity" {
  compartment_id      = var.compartment_ocid
  availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
  display_name        = "simcity-node"
  shape               = var.instance_shape

  shape_config {
    ocpus         = var.instance_ocpus
    memory_in_gbs = var.instance_memory_gb
  }

  create_vnic_details {
    subnet_id        = oci_core_subnet.simcity.id
    assign_public_ip = true
  }

  source_details {
    source_type             = "image"
    source_id               = data.oci_core_images.ubuntu.images[0].id
    boot_volume_size_in_gbs = var.boot_volume_gb
  }

  metadata = {
    ssh_authorized_keys = var.ssh_public_key
    # Bootstrap k3s so infra/k8s manifests can be applied immediately.
    user_data = base64encode(<<-CLOUDINIT
      #!/bin/bash
      set -euo pipefail
      curl -sfL https://get.k3s.io | sh -
      # Make kubeconfig readable for the default 'ubuntu' user.
      mkdir -p /home/ubuntu/.kube
      cp /etc/rancher/k3s/k3s.yaml /home/ubuntu/.kube/config
      chown -R ubuntu:ubuntu /home/ubuntu/.kube
    CLOUDINIT
    )
  }
}
