# 🚀 Deployment Guide - BNP Paribas Assistant Backend

## ✅ Current Deployment Status

**Production API:** http://bnp-assistant-api-oussama.eastus.azurecontainer.io:8000

- ✅ Deployed on Azure Container Instances (ACI)
- ✅ Using Azure Container Registry (ACR)
- ✅ Powered by OpenAI GPT-4
- ✅ 2 vCPU, 4GB RAM
- ✅ East US region

---

## 📋 Prerequisites

- [x] Docker Desktop installed and running (for local testing)
- [x] Azure account with Container Services enabled
- [x] Azure CLI installed
- [x] GitHub account with repository
- [x] OpenAI API key

---

## 🧪 Step 1: Test Locally with Frontend

### 1. Start Backend Container

```powershell
# Build image (if not already built)
docker build -t bnp-assistant:test .

# Start container
docker run -d --env-file .env -p 8000:8000 --name bnp-backend-test bnp-assistant:test

# Wait for initialization (check logs)
docker logs -f bnp-backend-test
```

### 2. Test Endpoints

```powershell
# Health check
curl http://localhost:8000/health

# Test query
curl -X POST http://localhost:8000/query `
  -H "Content-Type: application/json" `
  -d '{"question": "What is BNP Paribas?", "include_sources": true}'
```

### 3. Connect Frontend

Update your frontend `.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
# or for Vite
VITE_API_URL=http://localhost:8000
```

Start frontend and test!

### 4. Stop Container

```powershell
docker stop bnp-backend-test
docker rm bnp-backend-test
```

---

## ☁️ Step 2: Azure App Service Setup (One-Time)

### **Option A: Automated Script (Recommended)**

```powershell
# Run the deployment script
.\deploy-azure-appservice.ps1
```

This script will:
1. Login to Azure
2. Create Resource Group
3. Create App Service Plan (FREE B1 tier)
4. Create Web App
5. Configure settings
6. Get publish profile for GitHub

### **Option B: Manual Commands**

```powershell
# 1. Login
az login

# 2. Create Resource Group
az group create `
  --name bnp-assistant-rg `
  --location eastus

# 3. Create App Service Plan (FREE B1)
az appservice plan create `
  --name bnp-assistant-plan `
  --resource-group bnp-assistant-rg `
  --sku B1 `
  --is-linux

# 4. Create Web App
az webapp create `
  --resource-group bnp-assistant-rg `
  --plan bnp-assistant-plan `
  --name bnp-assistant-api `
  --runtime "PYTHON:3.11" `
  --startup-file "uvicorn rag_openai_prod:app --host 0.0.0.0 --port 8000"

# 5. Configure App Settings
az webapp config appsettings set `
  --resource-group bnp-assistant-rg `
  --name bnp-assistant-api `
  --settings `
    OPENAI_API_KEY="your-key-here" `
    WEBSITES_PORT="8000" `
    SCM_DO_BUILD_DURING_DEPLOYMENT="true"
```

**Cost:** **FREE** for 12 months with Azure Student! 🎉

---

## 🔐 Step 3: GitHub Secrets Setup

You need to add these secrets to your GitHub repository:

### 1. Create Azure Service Principal

```powershell
# Create service principal
az ad sp create-for-rbac `
  --name "bnp-assistant-github" `
  --role contributor `
  --scopes /subscriptions/YOUR_SUBSCRIPTION_ID/resourceGroups/bnp-assistant-rg `
  --sdk-auth
```

This outputs JSON like:
```json
{
  "clientId": "xxx",
  "clientSecret": "xxx",
  "subscriptionId": "xxx",
  "tenantId": "xxx",
  ...
}
```

### 2. Add GitHub Secrets

Go to: **GitHub Repo → Settings → Secrets and variables → Actions → New repository secret**

Add these secrets:

| Secret Name | Value |
|-------------|-------|
| `AZURE_CREDENTIALS` | Paste the entire JSON from step 1 |
| `OPENAI_API_KEY` | Your OpenAI API key (sk-...) |

---

## 🚀 Step 4: Deploy via GitHub Actions

### 1. Push Code to GitHub

```powershell
git add .
git commit -m "Add Azure deployment CI/CD"
git push origin main
```

### 2. Watch Deployment

- Go to: **GitHub Repo → Actions**
- Click on the running workflow
- Watch each step complete

### 3. Get Your Backend URL

After successful deployment:
- Check the workflow summary for the URL
- It will be: `http://bnp-assistant-api.REGION.azurecontainer.io:8000`

---

## ✅ Step 5: Verify Deployment

### Test Health

```powershell
curl http://bnp-assistant-api.westeurope.azurecontainer.io:8000/health
```

### Test Query

```powershell
curl -X POST http://bnp-assistant-api.westeurope.azurecontainer.io:8000/query `
  -H "Content-Type: application/json" `
  -d '{"question": "What services does BNP Paribas offer?", "include_sources": true}'
```

---

## 🔧 Step 6: Update Frontend for Production

Update your frontend `.env.production`:

```env
NEXT_PUBLIC_API_URL=http://bnp-assistant-api.westeurope.azurecontainer.io:8000
# or
VITE_API_URL=http://bnp-assistant-api.westeurope.azurecontainer.io:8000
```

Redeploy frontend to Vercel!

---

## 📊 Monitoring & Debugging

### View Container Logs

```powershell
az container logs `
  --resource-group bnp-assistant-rg `
  --name bnp-assistant-container
```

### Check Container Status

```powershell
az container show `
  --resource-group bnp-assistant-rg `
  --name bnp-assistant-container `
  --query "{Status:instanceView.state, FQDN:ipAddress.fqdn, CPU:containers[0].instanceView.currentState.detailStatus}"
```

### Restart Container

```powershell
az container restart `
  --resource-group bnp-assistant-rg `
  --name bnp-assistant-container
```

---

## 💰 Cost Breakdown (Azure Student)

| Service | Specs | Monthly Cost | From Credit |
|---------|-------|--------------|-------------|
| Container Registry | Basic | $5 | ✅ |
| Container Instance | 2 CPU, 4GB RAM | ~$73 (if 24/7) | ✅ |
| **Stop when not needed** | (Stop/Start) | ~$10-20/month | ✅ |

**Your $100 credit lasts: 5-10 months** with smart usage!

### Start/Stop Container to Save Money

```powershell
# Stop (saves money)
az container stop `
  --resource-group bnp-assistant-rg `
  --name bnp-assistant-container

# Start (when needed)
az container start `
  --resource-group bnp-assistant-rg `
  --name bnp-assistant-container
```

---

## 🔄 Update Deployment

Every time you push to `main` branch:
1. GitHub Actions automatically builds new Docker image
2. Pushes to Azure Container Registry
3. Deletes old container
4. Creates new container with latest code
5. Runs health checks

**No manual work needed!** 🎉

---

## 🧹 Cleanup (Delete Everything)

```powershell
# Delete entire resource group (careful!)
az group delete --name bnp-assistant-rg --yes --no-wait
```

---

## 📞 Useful Commands

```powershell
# List all containers
az container list --resource-group bnp-assistant-rg --output table

# Get container URL
az container show `
  --resource-group bnp-assistant-rg `
  --name bnp-assistant-container `
  --query ipAddress.fqdn -o tsv

# Stream logs
az container attach `
  --resource-group bnp-assistant-rg `
  --name bnp-assistant-container

# Check deployment history
az deployment group list `
  --resource-group bnp-assistant-rg `
  --output table
```

---

## 🤖 GitHub Actions CI/CD Setup

### Step 1: Create Azure Service Principal

Run this command to create credentials for GitHub Actions:

```powershell
az ad sp create-for-rbac `
  --name "github-actions-bnp-assistant" `
  --role contributor `
  --scopes /subscriptions/YOUR_SUBSCRIPTION_ID/resourceGroups/bnp-assistant-rg `
  --sdk-auth
```

**Get your subscription ID:**
```powershell
az account show --query id -o tsv
```

**Copy the entire JSON output** - it will look like this:
```json
{
  "clientId": "xxxxx",
  "clientSecret": "xxxxx",
  "subscriptionId": "xxxxx",
  "tenantId": "xxxxx",
  ...
}
```

### Step 2: Add GitHub Secrets

Go to your GitHub repository:
1. **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Add these two secrets:

| Secret Name | Value | Description |
|-------------|-------|-------------|
| `AZURE_CREDENTIALS` | The entire JSON from Step 1 | Azure login credentials |
| `OPENAI_API_KEY` | `sk-...` | Your OpenAI API key |

### Step 3: Test CI/CD

**Option 1: Manual Trigger**
1. Go to **Actions** tab in GitHub
2. Select "Deploy to Azure Container Instance"
3. Click **Run workflow** → **Run workflow**

**Option 2: Automatic Trigger**
1. Make any change to files in `Backend_server/` folder
2. Commit and push to `main` branch
3. GitHub Actions will automatically deploy!

```powershell
git add .
git commit -m "test: trigger CI/CD deployment"
git push origin main
```

### Step 4: Monitor Deployment

1. Go to **Actions** tab in GitHub
2. Click on the running workflow
3. Watch the deployment steps in real-time
4. Check the **Summary** for deployment URL

### What Happens During CI/CD?

```
📥 Checkout code
  ↓
🔐 Azure Login
  ↓
🐳 Login to ACR
  ↓
🏗️ Build Docker image (tagged with commit SHA + latest)
  ↓
⬆️ Push to Azure Container Registry
  ↓
🗑️ Delete old container
  ↓
🚀 Deploy new container to ACI
  ↓
⏳ Wait for startup (60s)
  ↓
🏥 Health check
  ↓
✅ Deployment complete!
```

### Troubleshooting CI/CD

**If deployment fails:**

1. **Check GitHub Actions logs**
   - Go to Actions tab
   - Click on failed workflow
   - Expand failed step

2. **Verify secrets are set**
   ```powershell
   # Test AZURE_CREDENTIALS locally
   az login --service-principal `
     --username CLIENT_ID `
     --password CLIENT_SECRET `
     --tenant TENANT_ID
   ```

3. **Check Azure permissions**
   ```powershell
   az role assignment list `
     --assignee YOUR_SERVICE_PRINCIPAL_CLIENT_ID `
     --resource-group bnp-assistant-rg
   ```

4. **View container logs**
   ```powershell
   az container logs `
     --resource-group bnp-assistant-rg `
     --name bnp-assistant-container
   ```

### Rollback to Previous Version

If a deployment breaks something, you can rollback:

```powershell
# List all images with their commit SHAs
az acr repository show-tags `
  --name bnpassistantacr `
  --repository bnp-assistant `
  --output table

# Deploy specific version
az container create `
  --resource-group bnp-assistant-rg `
  --name bnp-assistant-container `
  --image bnpassistantacr.azurecr.io/bnp-assistant:COMMIT_SHA `
  --cpu 2 --memory 4 --os-type Linux `
  --dns-name-label bnp-assistant-api-oussama `
  --ports 8000 `
  --environment-variables OPENAI_API_KEY="your-key" `
  --location eastus
```

---

## 🎉 You're Done!

Your backend is now:
- ✅ Containerized with Docker
- ✅ Deployed to Azure Container Instances
- ✅ Automated CI/CD with GitHub Actions
- ✅ Auto-deploying via GitHub Actions
- ✅ Monitored and logged
- ✅ Production-ready!

**Questions?** Check the workflow logs on GitHub Actions!
