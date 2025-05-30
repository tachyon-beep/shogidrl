# Shogi DRL Model Architecture Selection

This document outlines the process undertaken to select a suitable neural network architecture for the Keisei Deep Reinforcement Learning (DRL) Shogi agent.

## Background

The initial neural network architecture proposed by an LLM for the Shogi agent was deemed unlikely to possess the representational power required for effective learning in such a complex game. A more robust and specialized architecture was necessary.

## Architecture Selection Process

To identify a more appropriate model, the following steps were taken:

1.  **Sourcing Diverse Proposals:** Multiple Large Language Models (ChatGPT, DeepSeek, and Gemini) were prompted to propose neural network architectures specifically tailored for Shogi DRL.
2.  **Research-Driven Down-Selection:** The proposed architectures were then subjected to a comparative analysis. Gemini Deep Research capabilities were utilized to evaluate the strengths and weaknesses of each proposal, leading to a final selection.

The detailed comparison and the down-selection rationale can be found in the [Comparative Overview of Proposed Shogi DRL Architectures](deep_research.md).

## Outcome and Implementation

The selected architecture, chosen for its potential to effectively learn Shogi strategy, will serve as the blueprint for implementation. The next phase involves translating this design into functional code, a task intended for GitHub Copilot.

## Supporting Materials

-   **Detailed Analysis:** [Comparative Overview of Proposed Shogi DRL Architectures](deep_research.md)
-   **Podcast Version:** For an alternative way to consume the technical evaluation that led to this selection, an audio podcast version of the analysis is also available. *(You may want to specify where this podcast can be accessed if it's hosted or provide a direct link if it's a local file within the project structure).*
