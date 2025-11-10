# 🔐 GitHub Secrets Setup Guide

Quick guide to configure GitHub Actions CI/CD for automatic deployments.

---

## 📋 What You Need

1. ✅ Azure account (already done)
2. ✅ GitHub repository (already done)
3. ✅ OpenAI API key (already have)
4. 🆕 Azure Service Principal (need to create)

---

## 🚀 Step-by-Step Setup

### Step 1: Get Your Azure Subscription ID

```powershell
az account show --query id -o tsv
```

**Example output:** `722baa66-8c78-4ddc-8205-f145e25c5c64`

Copy this ID - you'll need it in Step 2.

---

### Step 2: Create Service Principal

Replace `YOUR_SUBSCRIPTION_ID` with the ID from Step 1:

```powershell
az ad sp create-for-rbac `
  --name "github-actions-bnp-assistant" `
  --role contributor `
  --scopes /subscriptions/YOUR_SUBSCRIPTION_ID/resourceGroups/bnp-assistant-rg `
  --sdk-auth
```

**Example:**
```powershell
az ad sp create-for-rbac `
  --name "github-actions-bnp-assistant" `
  --role contributor `
  --scopes /subscriptions/722baa66-8c78-4ddc-8205-f145e25c5c64/resourceGroups/bnp-assistant-rg `
  --sdk-auth
```

**Output will be JSON like this:**
```json
{
  "clientId": "12345678-1234-1234-1234-123456789abc",
  "clientSecret": "abcdefghijklmnopqrstuvwxyz123456789",
  "subscriptionId": "722baa66-8c78-4ddc-8205-f145e25c5c64",
  "tenantId": "87654321-4321-4321-4321-987654321xyz",
  "activeDirectoryEndpointUrl": "https://login.microsoftonline.com",
  "resourceManagerEndpointUrl": "https://management.azure.com/",
  "activeDirectoryGraphResourceId": "https://graph.windows.net/",
  "sqlManagementEndpointUrl": "https://management.core.windows.net:8443/",
  "galleryEndpointUrl": "https://gallery.azure.com/",
  "managementEndpointUrl": "https://management.core.windows.net/"
}
```

**⚠️ COPY THIS ENTIRE JSON OUTPUT** - You'll paste it into GitHub in Step 3!

---

### Step 3: Add Secrets to GitHub

#### 3.1 Navigate to Secrets Settings

1. Go to your GitHub repository: https://github.com/oussama95boussaid/-BNP-Paribas-virtuel-assistant-Chatbot
2. Click **Settings** (top menu)
3. Click **Secrets and variables** (left sidebar)
4. Click **Actions**
5. Click **New repository secret**

#### 3.2 Add AZURE_CREDENTIALS Secret

- **Name:** `AZURE_CREDENTIALS`
- **Value:** Paste the **entire JSON output** from Step 2
- Click **Add secret**

#### 3.3 Add OPENAI_API_KEY Secret

- Click **New repository secret** again
- **Name:** `OPENAI_API_KEY`
- **Value:** Your OpenAI API key (starts with `sk-...`)
- Click **Add secret**

**You should now have 2 secrets:**

| Secret Name | Description | Status |
|-------------|-------------|--------|
| `AZURE_CREDENTIALS` | Azure login credentials (JSON) | ✅ Required |
| `OPENAI_API_KEY` | OpenAI API key for GPT-4 | ✅ Required |

---

## ✅ Verify Setup

### Test the CI/CD Pipeline

**Option 1: Manual Trigger (Safest)**

1. Go to **Actions** tab in GitHub
2. Select "Deploy to Azure Container Instance" workflow
3. Click **Run workflow** dropdown
4. Click **Run workflow** button
5. Watch the deployment happen! 🚀

**Option 2: Automatic Trigger**

Make any small change and push:

```powershell
# Make a test change
echo "# Test CI/CD" >> README.md

# Commit and push
git add README.md
git commit -m "test: verify CI/CD pipeline"
git push origin main
```

Go to **Actions** tab to watch the deployment.

---

## 🔍 Expected Results

**If successful, you'll see:**

```
✅ Checkout code
✅ Azure Login
✅ Login to ACR
✅ Build and push Docker image
✅ Get ACR credentials
✅ Delete old container instance
✅ Deploy to Azure Container Instance
✅ Get container URL
✅ Wait for container startup
✅ Health check
✅ Deployment summary
```

**Final output:**
```
============================================
✅ DEPLOYMENT SUCCESSFUL!
============================================
Image: bnpassistantacr.azurecr.io/bnp-assistant:latest
Git SHA: abc123def456...
Container URL: http://bnp-assistant-api-oussama.eastus.azurecontainer.io:8000
Deployed to: eastus
============================================
```

---

## 🐛 Troubleshooting

### "Authentication failed" Error

**Problem:** Invalid Azure credentials

**Solution:**
1. Verify subscription ID is correct:
   ```powershell
   az account show --query id -o tsv
   ```
2. Re-create service principal with correct subscription ID
3. Update `AZURE_CREDENTIALS` secret in GitHub

### "OPENAI_API_KEY not found" Error

**Problem:** Missing or incorrect OpenAI API key

**Solution:**
1. Get your API key from: https://platform.openai.com/api-keys
2. Verify it starts with `sk-`
3. Update `OPENAI_API_KEY` secret in GitHub

### "Resource not found" Error

**Problem:** Resource group or ACR doesn't exist

**Solution:**
1. Verify resource group exists:
   ```powershell
   az group show --name bnp-assistant-rg
   ```
2. Verify ACR exists:
   ```powershell
   az acr show --name bnpassistantacr
   ```

### Health Check Fails

**Problem:** Container starts but health check returns error

**Solution:**
1. Check container logs in GitHub Actions output
2. Or manually check:
   ```powershell
   az container logs `
     --resource-group bnp-assistant-rg `
     --name bnp-assistant-container
   ```

---

## 🎯 What Happens Now?

**Every time you push to `main` branch:**

1. 🤖 GitHub Actions automatically triggers
2. 🐳 Builds new Docker image with your changes
3. ⬆️ Pushes to Azure Container Registry
4. 🗑️ Deletes old container
5. 🚀 Deploys new container
6. 🏥 Runs health checks
7. ✅ Updates production API automatically!

**Your changes go live in ~5-10 minutes with ZERO manual work!** 🎉

---

## 🔒 Security Best Practices

✅ **Never commit secrets to Git**
- Use `.env` file (already in `.gitignore`)
- Use GitHub Secrets (done!)
- Use Azure Key Vault for production (optional)

✅ **Rotate credentials periodically**
```powershell
# Create new service principal every 6 months
az ad sp create-for-rbac --name "github-actions-bnp-assistant-$(Get-Date -Format 'yyyyMM')" ...
```

✅ **Use minimal permissions**
- Service principal only has access to `bnp-assistant-rg` resource group
- No access to other Azure resources ✅

---

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Azure Service Principal Guide](https://learn.microsoft.com/en-us/cli/azure/create-an-azure-service-principal-azure-cli)
- [Azure Container Instances](https://learn.microsoft.com/en-us/azure/container-instances/)

---

**Need help?** Open an issue in the GitHub repository! 🤝
