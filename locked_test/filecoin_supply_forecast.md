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
under two scenarios that reflect the current trajectory of declining storage
provider (SP) onboarding.

The central finding is that **locked FIL declines under both scenarios**, with
the severity depending on whether renewal rates hold steady or deteriorate.
Circulating supply approaches 1 billion FIL in both cases. **Based on the
trajectory of the past year, the Decline scenario is the more realistic
projection.**

\bigskip

| Metric | Current | Status Quo | Decline |
|:-------|--------:|---------:|--------:|
| Network Locked (M FIL) | 101.7 | 71.6 | 60.0 |
| Circulating Supply (M FIL) | 836.9 | 951.2 | 962.1 |
| Locked / Circ Supply | 0.122 | 0.075 | 0.062 |
| Pledge per 32 GiB Sector (FIL) | 0.172 | 0.282 | 0.429 |
| 1Y Forward FoFR (%) | 23.1 | 28.9 | 54.8 |

Table: Forecast summary. "Current" reflects on-chain values as of February 22, 2026. Status Quo and Decline columns show values at the end of the two-year forecast horizon.

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

- Raw byte onboarding: 1.01 PiB/day (median)
- Renewal rate: 72.8% (mean)
- FIL+ rate: 97.1% (median)

Both scenarios share the same onboarding trajectory: a 3x per year exponential
decay, which matches the observed decline over the past 12 months. Both
scenarios also assume declining renewal rates, reflecting the expectation that
rising pledge requirements will deter marginal SPs from renewing. The scenarios
differ in the severity of this renewal rate erosion.

Both renewal rate curves follow a front-loaded (concave) shape: the decline is
steepest in the early period, reflecting the expectation that marginal SPs ---
those most sensitive to rising costs --- exit first, with the rate of decline
flattening as only committed operators remain.

**Status Quo** --- Onboarding continues its observed decline (3x/yr). The
renewal rate experiences mild erosion, declining from 73% to 55% over two
years. This represents the case where current conditions persist: no new growth
materializes, and rising pledge requirements gradually push out the least
committed SPs.

**Decline** --- Same onboarding trajectory, but renewal rate erosion is more
severe, declining from 73% to 36% over two years. This scenario reflects
accelerating SP attrition as the negative feedback loop between network
contraction and rising pledge requirements intensifies.

| Parameter | Status Quo | Decline |
|:----------|:---------|:--------|
| RB Onboarding | 1.01 → 0.34 (1Y) → 0.11 (2Y) PiB/d | Same |
| Renewal Rate | 73% → 60% (1Y) → 55% (2Y) | 73% → 47% (1Y) → 36% (2Y) |
| FIL+ Rate | 97.1% (flat) | 97.1% (flat) |

Table: Scenario input parameters. Both scenarios share declining onboarding and front-loaded renewal rate curves.

# Forecast Results

![Two-year scenario forecast panel. Rows from top: inputs, network power and
minting, supply metrics, and SP economics. The vertical dotted line marks the
transition from historical data (solid gray) to the forecast period
(dashed, color-coded by scenario).](forecast_panel.png)

## Network Power

Under the Status Quo scenario, quality-adjusted power (QAP) declines from
18.2 EiB to approximately 11.8 EiB (-35%). Even with only mild renewal rate
erosion, the falling rate of new onboarding means the network cannot replace
expiring capacity fast enough to maintain its current size.

The Decline scenario shows QAP falling to approximately 6.4 EiB (-65%) as
the front-loaded decline in renewal rate compounds the onboarding shortfall.
Sectors that would have renewed under flat conditions increasingly exit as
rising pledge requirements make renewal uneconomic.

## Locked FIL and Circulating Supply

Locked FIL is the most consequential metric for circulating supply dynamics.

Under the **Status Quo**, locked FIL declines from 101.7M to 71.6M (-30%).
Circulating supply grows from 837M to 951M (+14%), driven by ongoing minting
and vesting. L/CS declines from 0.122 to 0.075 (-38%).

Under the **Decline** scenario, the trajectory is more severe:

| Horizon | QAP | Locked FIL | Circ Supply | L/CS |
|:--------|----:|-----------:|------------:|-----:|
| Start | 18.2 EiB | 101.7M | 836.9M | 0.122 |
| +6 months | -7% | -6% | +6% | -12% |
| +1 year | -17% | -11% | +9% | -19% |
| +18 months | -49% | -34% | +13% | -41% |
| +2 years | -65% | -41% | +15% | -49% |

Table: Decline scenario — cumulative change from forecast start.

- **First year (decline begins immediately):** The front-loaded drop in
  renewal rate means locked FIL begins declining from the start, reaching
  -11% after one year. The 360-day sector duration still creates some inertia
  --- QAP contracts 17% while locked FIL trails at -11% --- but the steeper
  early RR decline ensures there is no false plateau. L/CS declines 19%.

- **Second year (acceleration phase):** Sectors locked during the transition
  begin to expire. With the renewal rate now approaching 36%, roughly
  two-thirds of expiring sectors release their pledge. Locked FIL drops 41%
  while circulating supply continues to grow (+15%). L/CS falls 49%,
  compounded by both the declining numerator and growing denominator.

## SP Economics

Pledge per sector exhibits an inverse relationship with network size under
attrition. As QAP contracts, the consensus pledge component --- which scales
with the ratio of sector size to total network power --- increases. Under
Status Quo, pledge rises from 0.17 to 0.28 FIL per 32 GiB sector (+64%);
under Decline, it reaches 0.43 FIL (+150%). This creates a negative feedback
loop: rising pledge deters renewals, which shrinks the network, which raises
pledge further.

The 1-year forward rate of return (FoFR) rises to approximately 29% under
Status Quo and 55% under Decline, up from 23%. This is mechanical: fewer
sectors compete for block rewards that decline more slowly than network power.
This elevated return represents the equilibrium incentive for remaining
providers but may be insufficient to attract new entrants if accompanied by
adverse token price dynamics.

# Key Takeaways

1. **Locked FIL declines under both scenarios.** Under Status Quo with mild
   renewal rate erosion, locked FIL falls 30% over two years. Under
   accelerating attrition, the decline reaches 41%. There is no stability
   scenario --- both paths lead to contraction.

2. **The Decline scenario is the more realistic projection.** Raw byte
   onboarding has already declined ~3x over the past year, and rising pledge
   requirements create a negative feedback loop that is likely to erode
   renewal rates. The Decline scenario projects L/CS compressing from 0.122
   to 0.062 over two years.

3. **Locked FIL has significant inertia.** The 360-day sector duration creates
   a lag between declining conditions and their effect on locked FIL. Even in
   the Decline scenario, locked FIL is down only 11% after one year despite
   QAP already contracting 17%. The full impact materializes in the second
   year as sectors locked during the transition begin to expire without
   renewal.

4. **Both onboarding and renewal rate drive the decline, but renewals dominate
   the flow.** The daily volume of power flowing through renewals (expiring
   sectors being re-locked) is roughly 10x larger than new onboarding in power
   terms. Changes in renewal behavior therefore have an outsized effect on
   locked FIL relative to changes in new onboarding.

5. **Attrition is self-reinforcing in pledge terms.** As the network shrinks,
   pledge per sector rises (+64% under Status Quo, +150% under Decline), which
   may further discourage participation. FoFR also rises as a partial offset,
   but the net effect on SP decision-making depends on token price dynamics
   not modeled here.

6. **Circulating supply is bounded but approaching 1B FIL.** Under both
   scenarios, circulating supply grows to 950--962M FIL over two years.
   Both scenarios approach the 1 billion FIL threshold, a level with no
   protocol mechanism for reversal.

7. **These projections are likely still optimistic.** The model's renewal
   mechanism recalculates pledge at current conditions and retains the higher
   of original and recalculated pledge. In practice, SPs facing substantially
   higher pledge requirements may choose not to renew --- a behavioral response
   the model does not capture. The on-chain locked FIL decline rate is
   currently steeper than even the Decline scenario projects for its first 90
   days, suggesting the true trajectory may be worse than modeled here.

# Appendix: Model Notes

- **Data source:** Spacescope API (on-chain supply and power statistics)
- **History window:** 365 days (Feb 2025 -- Feb 2026)
- **Forecast horizon:** 730 days (Feb 2026 -- Feb 2028)
- **Onboarding trajectory:** 3x/yr exponential decay, matching the observed
  historical decline in raw byte onboarding over the past 12 months.
- **Renewal rate:** 30-day mean (72.8%). The mean is used rather than the
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
- **Renewal rate curve shape:** Both scenarios use front-loaded (concave)
  renewal rate curves rather than linear ramps. This reflects the expectation
  that marginal SPs --- those most sensitive to rising costs --- exit first,
  producing a steeper initial decline that flattens as only committed
  operators remain.
- **Why declining renewal rates rather than flat?** Previous forecasts held
  renewal rate constant at the observed 30-day mean or median. This was
  appropriate when the network was growing or stable, but Filecoin is now in a
  fundamentally different regime: onboarding is in sustained decline, pledge
  per sector is rising, and block rewards at current FIL prices are unlikely to
  cover operating expenses for many SPs. In this environment, extending a flat
  historical renewal rate produces trajectories that contradict on-chain
  reality. The declining renewal rate curves used here reflect the expectation
  that SPs will increasingly choose not to renew as the economics deteriorate.
- **Known model bias (optimistic):** The model's renewal mechanism
  recalculates pledge at current conditions and takes the maximum of original
  and recalculated pledge (matching protocol behavior). This creates a
  mechanical effect where renewed sectors lock *more* FIL than they originally
  did. For example, with 374k FIL/day in scheduled pledge expirations and a
  73% renewal rate: the 27% that don't renew release \textasciitilde101k
  FIL/day, but the 73% that do renew have their pledge recalculated upward by
  \textasciitilde37% (reflecting the increase in pledge per sector over the
  past year), adding \textasciitilde101k FIL/day in uplift --- nearly
  offsetting the entire expiration loss. New onboarding then pushes the net
  flow positive. In practice, the rising pledge is itself what deters renewals
  --- SPs who face substantially higher collateral requirements may simply
  choose to exit rather than pay. The model cannot capture this feedback: it
  applies the renewal rate *independently* of the pledge increase. The
  declining renewal rate curves used in our scenarios are our way of
  approximating this behavioral response exogenously. Even so, the on-chain
  locked FIL decline rate (-145k to -170k FIL/day) remains steeper than the
  model's Decline scenario first-90-day slope (-62k FIL/day). These
  projections should therefore be interpreted as an optimistic bound.
