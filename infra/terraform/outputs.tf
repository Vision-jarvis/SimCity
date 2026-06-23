# Outputs for the SimCity OCI deployment.

output "instance_public_ip" {
  description = "Public IP of the SimCity node."
  value       = oci_core_instance.simcity.public_ip
}

output "ssh_command" {
  description = "Convenience SSH command."
  value       = "ssh ubuntu@${oci_core_instance.simcity.public_ip}"
}

output "service_urls" {
  description = "User-facing service endpoints once k8s manifests are applied."
  value = {
    frontend = "http://${oci_core_instance.simcity.public_ip}:3000"
    api      = "http://${oci_core_instance.simcity.public_ip}:8000/docs"
    grafana  = "http://${oci_core_instance.simcity.public_ip}:3001"
    neo4j    = "http://${oci_core_instance.simcity.public_ip}:7474"
  }
}
