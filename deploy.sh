#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# 默认 usage 输出
usage() {
  echo "Usage: $0 [--ci] [--cd] docker | swarm | sync"
  echo "Examples:"
  echo "  $0 docker         # 同步 + 构建镜像 + 部署容器 (Compose)"
  echo "  $0 swarm          # 部署最终形态到 Docker Swarm (1x self-contained fusion)"
  echo "  $0 --ci docker    # 仅同步 + 构建镜像"
  echo "  $0 --cd docker    # 仅部署容器"
  echo "  $0 sync           # 仅同步项目文件"
}

CI_MODE=false
CD_MODE=false
SYNC_ONLY=false
ANSIBLE_DIR="./ansible"
TARGET=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --ci)
      CI_MODE=true
      shift
      ;;
    --cd)
      CD_MODE=true
      shift
      ;;
    sync)
      SYNC_ONLY=true
      shift
      ;;
    docker|swarm)
      TARGET="$1"
      shift
      ;;
    *)
      usage
      exit 1
      ;;
  esac
done

if [ "$SYNC_ONLY" = false ] && [ -z "$TARGET" ]; then
  echo "Error: 请指定部署目标 (docker) 或使用 sync"
  echo
  usage
  exit 1
fi

if [ "$SYNC_ONLY" = true ]; then
  echo "仅同步项目文件 (sync)..."
  ansible-playbook -i "$ANSIBLE_DIR/inventory.yml" "$ANSIBLE_DIR/site.yml" --tags "sync"
  exit 0
fi

TAGS="$TARGET"
if [ "$CI_MODE" = true ] && [ "$CD_MODE" = false ]; then
  TAGS="$TAGS,sync,ci"
  echo "仅同步 + 构建镜像 ($TARGET,sync,ci)..."
elif [ "$CI_MODE" = false ] && [ "$CD_MODE" = true ]; then
  TAGS="$TAGS,cd"
  echo "仅部署服务 ($TARGET,cd)..."
else
  TAGS="$TAGS,sync,ci,cd"
  echo "完整CI/CD流水线部署 ($TARGET,sync,ci,cd)..."
fi

EXTRA_ARGS=()
if [ "$TARGET" = "docker" ]; then
  EXTRA_ARGS=(--skip-tags swarm)
fi

if [ "$TARGET" = "swarm" ]; then
  echo "目标仅发布 qwen-tts-fusion 单实例；旧 Base 退役是验收后的独立步骤"
  EXTRA_ARGS=(--skip-tags docker)
  if [ "$CD_MODE" = true ] && [ "$CI_MODE" = false ]; then
    if [[ ! "${QWEN_FUSION_IMAGE:-}" =~ ^registry\.ttd/qwen-tts-fusion/fusion:(h-[a-f0-9]{12,64})(@sha256:[a-f0-9]{64})?$ ]] && \
       [[ ! "${QWEN_FUSION_IMAGE:-}" =~ ^registry\.ttd/qwen-tts-fusion/fusion@sha256:[a-f0-9]{64}$ ]]; then
      echo "CD requires QWEN_FUSION_IMAGE from this committed snapshot's immutable CI result." >&2
      exit 1
    fi
    EXTRA_ARGS+=(-e "ci_hash_image_name=$QWEN_FUSION_IMAGE")
  fi
fi

# Formal builds consume a detached, clean copy of one recorded commit.
# Invoke through the registered TTD launcher so docker_ci_cd resolves consistently.
if [ "$TARGET" = "swarm" ] && [ -z "${QWEN_SOURCE_SNAPSHOT:-}" ]; then
  if [ -n "$(git status --porcelain)" ]; then
    echo "Commit all participating changes before formal deployment." >&2
    exit 1
  fi
  source_commit=$(git rev-parse HEAD)
  snapshot_parent=$(mktemp -d /tmp/qwen-fusion-source.XXXXXX)
  snapshot="$snapshot_parent/source"
  git clone --quiet --no-hardlinks --no-checkout "$PWD" "$snapshot"
  git -C "$snapshot" checkout --quiet --detach "$source_commit"
  chmod -R a-w "$snapshot"
  echo "Formal source: $source_commit; detached snapshot: $snapshot"
  phases=()
  [ "$CI_MODE" = true ] && phases+=(--ci)
  [ "$CD_MODE" = true ] && phases+=(--cd)
  QWEN_SOURCE_SNAPSHOT="$source_commit" bash "$snapshot/deploy.sh" "${phases[@]}" swarm
  test "$(git -C "$snapshot" rev-parse HEAD)" = "$source_commit"
  test -z "$(git -C "$snapshot" status --porcelain)"
  echo "Verified source snapshot unchanged: $source_commit"
  exit 0
fi
if [ "$TARGET" = "swarm" ]; then
  test "$(git rev-parse HEAD)" = "$QWEN_SOURCE_SNAPSHOT"
  test -z "$(git status --porcelain)"
fi
ansible-playbook -i "$ANSIBLE_DIR/inventory.yml" "$ANSIBLE_DIR/site.yml" \
  --tags "$TAGS" "${EXTRA_ARGS[@]}"
