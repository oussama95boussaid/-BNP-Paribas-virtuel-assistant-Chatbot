# ============================================================================
# Azure Container Instance Deployment Script
# Using ACR + ACI (Recommended for production)
# ============================================================================

Write-Host "🐳 Azure Container Instance Deployment - BNP Assistant" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

$RESOURCE_GROUP = "bnp-assistant-rg"
$ACR_NAME = "bnpassistantacr"
$CONTAINER_NAME = "bnp-assistant-container"
$IMAGE_NAME = "bnp-assistant"
$DNS_LABEL = "bnp-assistant-api-oussama"
$LOCATION = "eastus"

# ============================================================================
# 1. Create Azure Container Registry
# ============================================================================
Write-Host "1️⃣  Creating Azure Container Registry..." -ForegroundColor Yellow

$acrExists = az acr show --name $ACR_NAME --resource-group $RESOURCE_GROUP 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✅ ACR already exists" -ForegroundColor Green
} else {
    Write-Host "   Creating ACR (this may take 2-3 minutes)..." -ForegroundColor Gray
    az acr create `
      --resource-group $RESOURCE_GROUP `
      --name $ACR_NAME `
      --sku Basic `
      --location $LOCATION | Out-Null
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ ACR created successfully" -ForegroundColor Green
    } else {
        Write-Host "   ❌ Failed to create ACR!" -ForegroundColor Red
        exit 1
    }
}

# ============================================================================
# 2. Login to ACR
# ============================================================================
Write-Host ""
Write-Host "2️⃣  Logging in to ACR..." -ForegroundColor Yellow
az acr login --name $ACR_NAME | Out-Null

if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✅ Logged in to ACR" -ForegroundColor Green
} else {
    Write-Host "   ❌ Failed to login to ACR!" -ForegroundColor Red
    exit 1
}

# ============================================================================
# 3. Tag and Push Docker Image
# ============================================================================
Write-Host ""
Write-Host "3️⃣  Pushing Docker image to ACR..." -ForegroundColor Yellow

$ACR_LOGIN_SERVER = "$ACR_NAME.azurecr.io"
$FULL_IMAGE_NAME = "$ACR_LOGIN_SERVER/${IMAGE_NAME}:latest"

# Check if local image exists
$localImage = docker images -q bnp-assistant:test
if ([string]::IsNullOrEmpty($localImage)) {
    Write-Host "   ⚠️  Local image not found. Building..." -ForegroundColor Yellow
    docker build -t bnp-assistant:test .
    if ($LASTEXITCODE -ne 0) {
        Write-Host "   ❌ Failed to build image!" -ForegroundColor Red
        exit 1
    }
}

# Tag image
Write-Host "   Tagging image..." -ForegroundColor Gray
docker tag bnp-assistant:test $FULL_IMAGE_NAME

# Push to ACR
Write-Host "   Pushing to ACR (this may take 5-10 minutes)..." -ForegroundColor Gray
docker push $FULL_IMAGE_NAME

if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✅ Image pushed to ACR" -ForegroundColor Green
} else {
    Write-Host "   ❌ Failed to push image!" -ForegroundColor Red
    exit 1
}

# ============================================================================
# 4. Get ACR Credentials
# ============================================================================
Write-Host ""
Write-Host "4️⃣  Getting ACR credentials..." -ForegroundColor Yellow

$ACR_USERNAME = az acr credential show --name $ACR_NAME --query username -o tsv
$ACR_PASSWORD = az acr credential show --name $ACR_NAME --query passwords[0].value -o tsv

if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✅ Credentials retrieved" -ForegroundColor Green
} else {
    Write-Host "   ❌ Failed to get credentials!" -ForegroundColor Red
    exit 1
}

# ============================================================================
# 5. Get OpenAI API Key
# ============================================================================
Write-Host ""
Write-Host "5️⃣  Getting OpenAI API key from .env..." -ForegroundColor Yellow

if (Test-Path ".env") {
    $envContent = Get-Content .env
    $OPENAI_KEY = ""
    foreach ($line in $envContent) {
        if ($line -match "^OPENAI_API_KEY=(.+)$") {
            $OPENAI_KEY = $matches[1].Trim('"').Trim("'")
            break
        }
    }
    
    if ([string]::IsNullOrEmpty($OPENAI_KEY)) {
        Write-Host "   ⚠️  OPENAI_API_KEY not found in .env!" -ForegroundColor Yellow
        $OPENAI_KEY = Read-Host "   Please enter your OpenAI API key"
    } else {
        Write-Host "   ✅ OpenAI API key found" -ForegroundColor Green
    }
} else {
    Write-Host "   ⚠️  .env file not found!" -ForegroundColor Yellow
    $OPENAI_KEY = Read-Host "   Please enter your OpenAI API key"
}

# ============================================================================
# 6. Delete Old Container (if exists)
# ============================================================================
Write-Host ""
Write-Host "6️⃣  Checking for existing container..." -ForegroundColor Yellow

$containerExists = az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "   Deleting old container..." -ForegroundColor Gray
    az container delete `
      --resource-group $RESOURCE_GROUP `
      --name $CONTAINER_NAME `
      --yes | Out-Null
    Write-Host "   ✅ Old container deleted" -ForegroundColor Green
} else {
    Write-Host "   ✅ No existing container" -ForegroundColor Green
}

# ============================================================================
# 7. Create Container Instance
# ============================================================================
Write-Host ""
Write-Host "7️⃣  Creating Azure Container Instance (this may take 3-5 minutes)..." -ForegroundColor Yellow

az container create `
  --resource-group $RESOURCE_GROUP `
  --name $CONTAINER_NAME `
  --image $FULL_IMAGE_NAME `
  --cpu 2 `
  --memory 4 `
  --os-type Linux `
  --registry-login-server $ACR_LOGIN_SERVER `
  --registry-username $ACR_USERNAME `
  --registry-password $ACR_PASSWORD `
  --dns-name-label $DNS_LABEL `
  --ports 8000 `
  --environment-variables OPENAI_API_KEY="$OPENAI_KEY" `
  --location $LOCATION | Out-Null

if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✅ Container instance created" -ForegroundColor Green
} else {
    Write-Host "   ❌ Failed to create container instance!" -ForegroundColor Red
    exit 1
}

# ============================================================================
# 8. Get Container URL
# ============================================================================
Write-Host ""
Write-Host "8️⃣  Getting container URL..." -ForegroundColor Yellow

$CONTAINER_FQDN = az container show `
  --resource-group $RESOURCE_GROUP `
  --name $CONTAINER_NAME `
  --query ipAddress.fqdn -o tsv

$CONTAINER_URL = "http://${CONTAINER_FQDN}:8000"

# ============================================================================
# 9. Summary
# ============================================================================
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "✅ DEPLOYMENT SUCCESSFUL!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Your Backend API:" -ForegroundColor Cyan
Write-Host "  URL: $CONTAINER_URL" -ForegroundColor White
Write-Host "  Health: ${CONTAINER_URL}/health" -ForegroundColor White
Write-Host "  Query: ${CONTAINER_URL}/query" -ForegroundColor White
Write-Host ""
Write-Host "Container Details:" -ForegroundColor Cyan
Write-Host "  Resource Group: $RESOURCE_GROUP" -ForegroundColor White
Write-Host "  Container: $CONTAINER_NAME" -ForegroundColor White
Write-Host "  Registry: $ACR_LOGIN_SERVER" -ForegroundColor White
Write-Host ""
Write-Host "Useful Commands:" -ForegroundColor Yellow
Write-Host "  View logs:     az container logs -g $RESOURCE_GROUP -n $CONTAINER_NAME" -ForegroundColor Gray
Write-Host "  Restart:       az container restart -g $RESOURCE_GROUP -n $CONTAINER_NAME" -ForegroundColor Gray
Write-Host "  Stop:          az container stop -g $RESOURCE_GROUP -n $CONTAINER_NAME" -ForegroundColor Gray
Write-Host "  Start:         az container start -g $RESOURCE_GROUP -n $CONTAINER_NAME" -ForegroundColor Gray
Write-Host ""
Write-Host "💰 Cost: ~$0.10/hour (stop when not needed to save money!)" -ForegroundColor Yellow
Write-Host ""
Write-Host "🎉 Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Wait 2-3 minutes for container to start" -ForegroundColor White
Write-Host "  2. Test: curl ${CONTAINER_URL}/health" -ForegroundColor White
Write-Host "  3. Update frontend with this URL" -ForegroundColor White
Write-Host ""
