# Reinforcement Learning: MDPs, Bellman, and SARSA

Lecture deck for a 3-hour BSc engineering course.

**Slides (live):** https://sml-fs26.github.io/classic_rl_teaching_material/

## Topics

- Markov Decision Processes (state, action, transition kernel, reward, discount)
- Markov property, policies, returns
- State-value $v_\pi$ and action-value $q_\pi$
- Bellman expectation equation (derivation + boxed forms)
- Bellman optimality (named, not dwelt on)
- Sample × bootstrap = Temporal Difference
- ε-greedy exploration, GLIE, schedules
- SARSA: derivation, pseudocode, hand-trace, convergence
- Windy Gridworld (canonical SARSA showcase)
- 5-minute Q-learning teaser ending on Cliff Walking

## Build

Source: `slides.qmd` (Quarto + reveal.js). Render with:

```sh
quarto render slides.qmd
```

Print-PDF (for handouts):

```sh
chromium --headless --print-to-pdf=slides_print.pdf "file://$PWD/slides.html?print-pdf"
```
