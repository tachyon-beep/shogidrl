# GitHub Actions Workflows

This directory contains the CI/CD workflows for the Keisei Deep Reinforcement Learning project.

## Active Workflows

### 🤖 Claude Code (`claude.yml`)
- **Triggers**: When `@claude` is mentioned in issues, PR comments, or reviews
- **Purpose**: Interactive AI assistance for development tasks
- **Tools Available**: pytest, train.py, evaluation, linting, type checking
- **Context**: Specialized for Keisei's manager-based RL architecture

### 📝 Claude Code Review (`claude-code-review.yml`) 
- **Triggers**: When PRs are opened/updated with Python code changes
- **Purpose**: Automated code review focusing on RL best practices
- **Scope**: 
  - Manager architecture adherence
  - Shogi game logic correctness
  - PyTorch performance patterns
  - Deep RL training stability

## Disabled Workflows

### ❌ CI Pipeline (`ci.yml.disabled`)
- **Status**: Temporarily disabled to focus on Claude-only workflows
- **Contains**: Full test suite, linting, security scans, performance checks
- **Note**: Can be re-enabled by renaming to `ci.yml` when needed

## Workflow Configuration

Both Claude workflows are customized for the Keisei project:
- **File Filtering**: Only runs on relevant Python files and configs
- **Project Context**: Understands RL training, Shogi rules, and architecture patterns  
- **Allowed Commands**: Limited to safe development and testing operations
- **Review Focus**: Emphasizes training stability and production readiness

## Setup Requirements

The Claude workflows require:
- `CLAUDE_CODE_OAUTH_TOKEN` secret configured in repository settings
- Appropriate permissions for reading issues, PRs, and repository content