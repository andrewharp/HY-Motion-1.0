#!/bin/bash
# GitHub Actions Workflow Watcher - waits for workflow without sleep loops
set -e

WORKFLOW_FILE="${1:-docker-build.yml}"
BRANCH="${2:-$(git branch --show-current)}"
TIMEOUT="${3:-1800}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

if ! command -v gh &> /dev/null; then log_error "GitHub CLI (gh) required"; exit 1; fi
if ! gh auth status &> /dev/null; then log_error "Run: gh auth login"; exit 1; fi

REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || echo "")
[ -z "$REPO" ] && { log_error "Not in git repo"; exit 1; }

log_info "Waiting for: $WORKFLOW_FILE on $BRANCH"
log_info "Repo: $REPO"

START_TIME=$(date +%s)
LAST_RUN_ID=""
INTERVAL=5

while true; do
    ELAPSED=$(($(date +%s) - START_TIME))
    [ $ELAPSED -ge $TIMEOUT ] && { log_error "Timeout after ${TIMEOUT}s"; exit 1; }
    
    RUN_INFO=$(gh run list --repo "$REPO" --workflow "$WORKFLOW_FILE" --branch "$BRANCH" --limit 1 --json databaseId,status,conclusion,url -q '.[0]' 2>/dev/null)
    [ -z "$RUN_INFO" ] && { sleep $INTERVAL; continue; }
    
    RUN_ID=$(echo "$RUN_INFO" | jq -r '.databaseId')
    STATUS=$(echo "$RUN_INFO" | jq -r '.status')
    CONCLUSION=$(echo "$RUN_INFO" | jq -r '.conclusion')
    URL=$(echo "$RUN_INFO" | jq -r '.url')
    
    [ "$RUN_ID" != "$LAST_RUN_ID" ] && { [ -n "$LAST_RUN_ID" ] && log_info "New run: $RUN_ID"; LAST_RUN_ID=$RUN_ID; }
    
    case "$STATUS" in
        completed)
            echo ""
            if [ "$CONCLUSION" = "success" ]; then
                log_info "✓ Workflow succeeded!"
                log_info "URL: $URL"
                exit 0
            else
                log_error "✗ Workflow failed: $CONCLUSION"
                log_error "URL: $URL"
                exit 1
            fi
            ;;
        queued) printf "\r${YELLOW}[QUEUED]${NC} Waiting for runner... %ds   " $ELAPSED ;;
        in_progress) printf "\r${BLUE}[RUNNING]${NC} Workflow running... %ds     " $ELAPSED ;;
    esac
    
    INTERVAL=$((INTERVAL < 30 ? INTERVAL + 2 : 30))
    sleep $INTERVAL &
    SLEEP_PID=$!
    wait $SLEEP_PID 2>/dev/null || true
done
