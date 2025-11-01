# ============================================================================
# Azure Deployment Guide - Quick Commands Reference
# ============================================================================

# STEP 1: Login to Azure
az login

# STEP 2: Set your subscription (if you have multiple)
az account set --subscription "Azure for Students"

# STEP 3: Create Resource Group
az group create \
  --name bnp-assistant-rg \
  --location westeurope

# STEP 4: Create Azure Container Registry (ACR)
az acr create \
  --resource-group bnp-assistant-rg \
  --name bnpassistantacr \
  --sku Basic

# STEP 5: Login to ACR
az acr login --name bnpassistantacr

# STEP 6: Build and push Docker image to ACR
az acr build \
  --registry bnpassistantacr \
  --image bnp-assistant:v1 \
  --file Dockerfile \
  .

# STEP 7: Get ACR credentials
$ACR_USERNAME = az acr credential show --name bnpassistantacr --query username -o tsv
$ACR_PASSWORD = az acr credential show --name bnpassistantacr --query passwords[0].value -o tsv

# STEP 8: Create Azure Container Instance with your OpenAI key
az container create \
  --resource-group bnp-assistant-rg \
  --name bnp-assistant-api \
  --image bnpassistantacr.azurecr.io/bnp-assistant:v1 \
  --cpu 2 \
  --memory 4 \
  --registry-login-server bnpassistantacr.azurecr.io \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --dns-name-label bnp-assistant-api \
  --ports 8000 \
  --environment-variables OPENAI_API_KEY="YOUR_OPENAI_KEY_HERE"

# STEP 9: Get your API URL
az container show \
  --resource-group bnp-assistant-rg \
  --name bnp-assistant-api \
  --query ipAddress.fqdn -o tsv

# STEP 10: Test your deployed API
# The URL will be: http://bnp-assistant-api.westeurope.azurecontainer.io:8000
