# Fly Brain Pilot
### Connectome-Based Drone Avoidance

> **Status: Work in progress.** This project is in early development. The architecture, experiments, and capabilities described below are planned research goals, not completed or validated results.

## Overview

Fly Brain Pilot explores whether a fruit fly’s neural wiring can help a drone respond to incoming obstacles while continuing toward its destination.

We plan to simulate selected visual motion, looming detection, and escape circuits as a spiking neural network (SNN). A small trained readout will translate neural activity into avoidance commands, while a conventional flight controller handles stabilization.

Our broader goal is to investigate responsive, human-aware robotic navigation and determine whether biological connectivity offers advantages in reaction time, reliability, or computational efficiency.

## Planned Approach

- Use FlyWire and/or MaleCNS connectome data to model relevant neural circuits.
- Reproduce a published result with a leaky integrate-and-fire simulation before extending the model.
- Convert drone camera frames into inputs suitable for the fly-inspired visual system.
- Keep connectome wiring fixed and train a small readout to generate avoidance commands.
- Integrate the avoidance layer into CrazySim using the Crazyflie drone model.
- Explore a separate person detector to apply larger safety distances and slower speeds around simulated people.

The exact datasets, tools, and detector architecture may change as we evaluate feasibility and incorporate mentor feedback.

## Research Roadmap

- [ ] Review connectome simulation, fly vision, and drone avoidance literature.
- [ ] Set up the simulation environment and reproduce a published neural response.
- [ ] Identify visual motion, looming, and escape circuits.
- [ ] Develop and test camera-to-neural input conversion.
- [ ] Establish baseline waypoint flight in CrazySim.
- [ ] Train the avoidance readout using simulated flight data.
- [ ] Integrate obstacle avoidance and person detection.
- [ ] Compare against CNN and shuffled-connectome baselines.
- [ ] Evaluate robustness and document reproducible results.
- [ ] Consider controlled hardware testing if simulation results justify it.

These milestones describe our intended direction and will be updated as development progresses.

## Planned Evaluation

We will compare the proposed system against:

- A conventional CNN controller trained on comparable data.
- A shuffled connectome with randomized connectivity.
- Models with selected looming or escape circuits disabled.

We plan to measure collision rate, task completion, minimum distance to simulated people, reaction time, and simulation throughput. Tests will include unfamiliar obstacles, moving objects, lighting changes, camera noise, and environmental disturbances.

We do not assume the connectome-based approach will outperform the alternatives. We will document negative results and limitations alongside successful experiments.

## Future Goals and Safety

Our long-term ambition is to understand whether connectome-based computing can contribute to practical robotic navigation.

Testing will begin in simulation. All experiments involving people will remain simulated. If we later test a physical Crazyflie, experiments will take place in a controlled flight area without people, using objects as obstacle stand-ins.

Physical deployment, real-time performance, and demonstrated safety remain future goals.

## Reproducibility

As the project develops, we intend to document:

- Dataset versions and upstream sources.
- Simulation settings and neuron parameters.
- Training, validation, and test procedures.
- Experiment configurations, measurements, and limitations.
- Setup instructions and dependency requirements.

We will follow applicable licenses and citation requirements for dataset and software use.

## Team and Mentorship

**Team:** Adam Hart, Kevin Puebla, and Christian Maestas  
**Mentor(s):** Dr. Melanie Moses; Dr. Wenbin Wan; Felix Wang, Sandia National Laboratories

This repository will evolve as we refine the research scope, implement the simulation, and share our methods and findings.
