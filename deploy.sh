#!/bin/bash
# Deploy Lead Scoring System to GCP Cloud Run
# Usage: ./deploy.sh [PROJECT_ID]

set -e

PROJECT_ID=${1:-$(gcloud config get-value project)}
REGION="us-central1"

echo "🚀 Deploying Lead Scoring System to GCP"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"

# Enable required APIs
echo "📡 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com run.googleapis.com containerregistry.googleapis.com --project=$PROJECT_ID

# Submit build
echo "🔨 Building and deploying containers..."
gcloud builds submit --config cloudbuild.yaml --project=$PROJECT_ID --substitutions=_REGION=$REGION

# Get service URLs
echo "🔗 Getting service URLs..."
API_URL=$(gcloud run services describe lead-scoring-api --region=$REGION --project=$PROJECT_ID --format='value(status.url)')
DASHBOARD_URL=$(gcloud run services describe lead-scoring-dashboard --region=$REGION --project=$PROJECT_ID --format='value(status.url)')

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 Dashboard (Portfolio): $DASHBOARD_URL"
echo "🔌 API Endpoint: $API_URL"
echo ""
echo "Test the API:"
echo "  curl $API_URL/health"
echo ""
echo "Setup weekly Cloud Scheduler job:"
echo "  gcloud scheduler jobs create http lead-scoring-weekly \\"
echo "    --schedule='0 9 * * 1' \\"
echo "    --uri='$API_URL/score/batch' \\"
echo "    --http-method=POST \\"
echo "    --message-body='{}' \\"
echo "    --time-zone='America/New_York'"
