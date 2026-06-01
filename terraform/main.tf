terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "5.6.0"
    }
  }
}

provider "google" {
  project     = var.gcp_project_id
  region      = var.region
  credentials = file(var.credentials_path)
}

resource "google_storage_bucket" "data_lake" {
  name          = var.gcp_bucket_name
  location      = var.region
  force_destroy = true
  uniform_bucket_level_access = true
}