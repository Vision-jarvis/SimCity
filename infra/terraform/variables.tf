# Input variables for the SimCity Oracle Cloud Always-Free deployment.

variable "tenancy_ocid" {
  description = "OCID of your OCI tenancy."
  type        = string
}

variable "user_ocid" {
  description = "OCID of the OCI user running Terraform."
  type        = string
}

variable "fingerprint" {
  description = "Fingerprint of the API signing key."
  type        = string
}

variable "private_key_path" {
  description = "Path to the OCI API private key (PEM)."
  type        = string
}

variable "region" {
  description = "OCI region identifier (e.g. us-ashburn-1)."
  type        = string
  default     = "us-ashburn-1"
}

variable "compartment_ocid" {
  description = "OCID of the compartment to deploy into."
  type        = string
}

variable "ssh_public_key" {
  description = "SSH public key injected into the instance for access."
  type        = string
}

# Always-Free Ampere A1: up to 4 OCPU / 24 GB RAM total at no cost.
variable "instance_shape" {
  description = "Compute shape. VM.Standard.A1.Flex is Always-Free eligible."
  type        = string
  default     = "VM.Standard.A1.Flex"
}

variable "instance_ocpus" {
  description = "OCPUs for the flex shape (Always-Free allows up to 4)."
  type        = number
  default     = 4
}

variable "instance_memory_gb" {
  description = "Memory in GB for the flex shape (Always-Free allows up to 24)."
  type        = number
  default     = 24
}

variable "boot_volume_gb" {
  description = "Boot volume size in GB (Always-Free total block storage is 200 GB)."
  type        = number
  default     = 100
}
