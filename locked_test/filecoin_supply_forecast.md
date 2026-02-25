---
title: "Filecoin Network Supply Forecast: Scenario Analysis"
subtitle: "CEL — February 2026"
geometry: margin=1in
fontsize: 11pt
linestretch: 1.15
header-includes:
  - \usepackage{booktabs}
  - \usepackage{float}
  - \floatplacement{figure}{H}
---

# Executive Summary

This document presents a two-year forward simulation of the Filecoin network
under two scenarios that vary storage provider (SP) onboarding and renewal
behavior.

The central finding is that **even holding current conditions flat, locked FIL
and L/CS decline over the forecast horizon**. The Decline scenario, which
extrapolates recent trends in onboarding and renewal behavior, projects a more
significant contraction: locked FIL falls 39% while circulating supply grows
15%, compressing L/CS from 0.122 to 0.065. **Based on the trajectory of the
past year, the Decline scenario is the more realistic projection.**

\bigskip

| Metric | Current | Status Quo | Decline |
|:-------|--------:|---------:|--------:|
| Network Locked (M FIL) | 101.8 | 99.4 | 62.5 |
| Circulating Supply (M FIL) | 836.6 | 924.5 | 960.0 |
| Locked / Circ Supply | 0.122 | 0.108 | 0.065 |
| Pledge per 32 GiB Sector (FIL) | 0.172 | 0.162 | 0.360 |
| 1Y Forward FoFR (%) | 22.7 | 21.9 | 50.2 |

Table: Forecast summary. "Current" reflects on-chain values as of February 21, 2026. Status Quo and Decline columns show values at the end of the two-year forecast horizon.

# Implications and Recommendations

1. **Circulating supply is approaching 1 billion FIL.** Under both scenarios,
   circulating supply crosses 900M FIL within the forecast horizon and
   approaches 960M under Decline. With locked FIL contracting and minting
   ongoing, the network is on track to cross the 1B threshold. Once crossed,
   there is no protocol mechanism to reverse this --- the only path back is
   increased locking demand.

2. **The network needs an alternative locking mechanism urgently.** Sector
   pledge is the only meaningful source of FIL locking today, and it is
   shrinking. Implementing **veFIL** (or an equivalent vote-escrowed locking
   mechanism) would create a second source of locking demand that is not tied
   to storage onboarding. This should be treated as a high-priority initiative.

3. **It is worse for Filecoin if no one wants to onboard capacity than if there
   are no paid deals.** The onboarding decline is the more existential threat.
   Paid storage deals are desirable, but the network's economic security
   depends on SPs continuing to onboard and renew sectors regardless of deal
   status. Policy decisions should prioritize removing barriers to onboarding
   above all else.

4. **FIL+ should be reconsidered.** Filecoin no longer provides an effective
   subsidy to SPs --- even with paid deals, the economics are marginal. FIL+
   has become an administrative burden across the ecosystem (notaries,
   allocators, compliance) without delivering commensurate onboarding growth.
   We should strongly consider removing FIL+ entirely by granting all sectors
   the 10x quality multiplier by default. This would eliminate friction for SPs,
   reduce governance overhead, and acknowledge the reality that the
   verified-deal distinction no longer drives meaningful network growth.

# Scenario Definitions

The simulation is initialized with 365 days of on-chain history (February 2025
-- February 2026). Key inputs are derived from 30-day trailing statistics.

**Current network parameters (30-day):**

- Raw byte onboarding: 0.99 PiB/day (median)
- Renewal rate: 72.1% (mean)
- FIL+ rate: 97.1% (median)

Two scenarios are modeled:

**Status Quo** --- Current onboarding and renewal rates are held flat for the
duration of the forecast. This represents the optimistic case where current
conditions do not deteriorate further.

**Decline** --- New onboarding decays exponentially at 3x per year (falling to
one-third of its current level after 12 months and one-ninth after 24 months),
while the renewal rate declines linearly from 72% to 36% over the forecast
horizon. This scenario extrapolates the declining trends observed over the past
year in both SP onboarding and renewal behavior. Raw byte onboarding has
already declined approximately 3x over the past 12 months, and a continued
deterioration in SP participation would compound through both reduced new
entrants and increasing non-renewal of existing capacity.

| Parameter | Status Quo | Decline |
|:----------|:---------|:--------|
| RB Onboarding | 0.99 PiB/d (flat) | 0.99 → 0.33 (1Y) → 0.11 (2Y) PiB/d |
| Renewal Rate | 72.1% (flat) | 72% → 36% (linear) |
| FIL+ Rate | 97.1% (flat) | 97.1% (flat) |

Table: Scenario input parameters.

# Forecast Results

![Two-year scenario forecast panel. Rows from top: inputs, network power and
minting, supply metrics, and SP economics. The vertical dotted line marks the
transition from historical data (solid gray) to the forecast period
(dashed, color-coded by scenario).](forecast_panel.png)

## Network Power

Under the Status Quo scenario, quality-adjusted power (QAP) declines modestly
from 18.2 EiB to approximately 16.9 EiB (-7%). At current onboarding rates,
new capacity does not fully offset sector expiry, so the network slowly
contracts even with flat inputs. The 9.7x FIL+ multiplier amplifies the effect
of raw byte changes on QAP, so even modest differences in renewal behavior
produce significant QAP movements.

The Decline scenario shows QAP falling to approximately 7.5 EiB (-59%) as both
onboarding and renewals contract. The exponential decay in new onboarding
reduces the flow of new capacity, while the declining renewal rate accelerates
the release of expiring sectors.

## Locked FIL and Circulating Supply

Locked FIL is the most consequential metric for circulating supply dynamics.

Under the **Status Quo**, locked FIL declines modestly from 101.8M to 99.4M
(-2%). Circulating supply grows from 837M to 925M (+11%), driven by ongoing
minting and vesting. The net result is that L/CS declines from 0.122 to 0.108
(-11%).

Under the **Decline** scenario, the trajectory unfolds in two distinct phases:

| Horizon | QAP | Locked FIL | Circ Supply | L/CS |
|:--------|----:|-----------:|------------:|-----:|
| Start | 18.2 EiB | 101.8M | 836.6M | 0.122 |
| +6 months | -5% | -3% | +6% | -8% |
| +1 year | -12% | -5% | +9% | -13% |
| +18 months | -41% | -28% | +12% | -36% |
| +2 years | -59% | -39% | +15% | -47% |

Table: Decline scenario — cumulative change from forecast start.

- **First year (inertia phase):** Although QAP begins to contract more
  noticeably (-12%), locked FIL is only down 5%. This reflects the 360-day
  sector duration: pledge locked by renewing sectors today is not released until
  those sectors expire a year later. The renewal rate is still above 50% in
  this period, so a majority of expiring sectors re-lock. L/CS declines 13%,
  driven by both modest locked FIL contraction and circulating supply growth.

- **Second year (acceleration phase):** The cumulative effect of declining
  renewals becomes visible as sectors locked in the first year of the forecast
  begin to expire. With the renewal rate now approaching 36%, roughly two-thirds
  of expiring sectors release their pledge. Locked FIL drops 39% while
  circulating supply continues to grow (+15%) as released pledge and ongoing
  minting add to circulation. L/CS falls 47%, compounded by both the declining
  numerator and growing denominator.

## SP Economics

Pledge per sector exhibits an inverse relationship with network size under
attrition. As QAP contracts in the Decline scenario, the consensus pledge
component --- which scales with the ratio of sector size to total network power
--- increases from 0.17 to 0.36 FIL per 32 GiB sector, a 109% increase in
collateral requirements despite a shrinking network.

The 1-year forward rate of return (FoFR) rises in the Decline scenario to
approximately 50%, up from 23%. This is mechanical: fewer sectors compete for
block rewards that decline more slowly than network power. This elevated return
represents the equilibrium incentive for remaining providers but may be
insufficient to attract new entrants if accompanied by adverse token price
dynamics.

# Key Takeaways

1. **Even at flat conditions, the network is slowly contracting.** At current
   onboarding and renewal rates, locked FIL declines 2% and L/CS falls from
   0.122 to 0.108 over two years. The Status Quo scenario is not a growth
   scenario --- it is a slow-erosion scenario.

2. **The Decline scenario is the more realistic projection.** Raw byte
   onboarding has already declined ~3x over the past year. The Decline scenario
   extrapolates this trend and projects locked FIL falling 39% and L/CS
   compressing to 0.065 over two years.

3. **Locked FIL has significant inertia.** The 360-day sector duration creates
   a lag between declining conditions and their effect on locked FIL. In the
   Decline scenario, locked FIL is only down 5% after one year despite QAP
   already contracting 12%. The full impact materializes in the second year as
   sectors locked during the transition begin to expire without renewal.

4. **Both onboarding and renewal rate drive the decline, but renewals dominate
   the flow.** The daily volume of power flowing through renewals (expiring
   sectors being re-locked) is roughly 10x larger than new onboarding in power
   terms. Changes in renewal behavior therefore have an outsized effect on
   locked FIL relative to changes in new onboarding.

5. **Attrition is self-reinforcing in pledge terms.** As the network shrinks,
   pledge per sector rises (+109% in the Decline scenario), which may further
   discourage participation. FoFR also rises as a partial offset, but the net
   effect on SP decision-making depends on token price dynamics not modeled here.

6. **Circulating supply is bounded.** Even under the Decline scenario,
   circulating supply increases by only ~15% over two years (837M to 960M FIL).
   The supply trajectory is dominated by vesting and minting schedules, with
   pledge dynamics providing a secondary modulating effect.

# Appendix: Model Notes

- **Data source:** Spacescope API (on-chain supply and power statistics)
- **History window:** 365 days (Feb 2025 -- Feb 2026)
- **Forecast horizon:** 730 days (Feb 2026 -- Feb 2028)
- **Renewal rate:** 30-day mean (72.1%). The mean is used rather than the
  median (82.1%) because locked FIL is a cumulative stock: total pledge
  retained = $\sum_i \text{RR}_i \times \text{scheduled\_expiration}_i \approx
  \overline{\text{RR}} \times \sum_i \text{scheduled\_expiration}_i$.
  The mean captures the full distribution including low-RR days that the
  median would discard.
- **FIL+ rate:** Held constant at 97.1% across all scenarios.
- **Sector duration:** 360 days (average).
- **Initialization:** Pledge/reward split estimated from 180-day linear
  vesting integral of on-chain minting data (4.5% reward-locked, 95.5%
  pledge-locked).
- **Reward vesting approximation:** The model uses a balance/180 decay for
  reward release rather than per-cohort linear vesting. This introduces
  approximately 2--3M FIL of drift over the historical window, which is
  acceptable for forecasting purposes.
