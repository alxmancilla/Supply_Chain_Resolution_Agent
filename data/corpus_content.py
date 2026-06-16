"""Source documents for the knowledge_corpus.

Each entry is a logical document that will be split into chunks by
seed_corpus.py (one chunk per paragraph). Total chunks land around 100.
"""
from __future__ import annotations

DOCUMENTS: list[dict] = [
    # ---------------- Carrier SLAs ----------------
    {
        "doc_type": "carrier_sla",
        "source": "carrier_agreements/carrier_a_2026.pdf",
        "metadata": {"carrier": "Carrier A", "effective_date": "2026-01-01"},
        "paragraphs": [
            "Carrier A SLA — Master Service Agreement effective 2026-01-01. Carrier A operates a dedicated dry-van fleet across the southern US corridor with full coverage of the TX-AZ, TX-TX, and TX-NM lanes. Standard transit on TX-AZ Austin to Phoenix is 36 hours door-to-door.",
            "Surcharge structure: Carrier A applies no fuel surcharge to shipments at or below the per-lane weight threshold (22,000 lbs on TX-AZ, 20,000 lbs on TX-TX). Above the threshold, a flat 4.5% fuel surcharge is applied to linehaul. There are no accessorial surcharges and no peak-season multiplier on intra-Texas moves.",
            "On-time performance guarantee: 98.2% on-time delivery on TX-AZ. Late deliveries beyond 4 hours past the appointment window trigger a 5% linehaul credit, applied automatically on the next invoice.",
            "Capacity commitment: 12 weekly truckloads guaranteed on TX-AZ and 20 on TX-TX, bookable up to 5 business days in advance. Spot capacity available with 48 hours notice subject to availability.",
            "Equipment: 53' dry vans, GPS-tracked, with electronic logging device (ELD) compliance. Reefer trailers available on TX-AZ at a flat $0.18/mile premium. Driver assist for loading/unloading is included up to two hours.",
            "Claims process: damage claims must be filed within 9 business days of delivery. Carrier A self-insures up to $100,000 per shipment with no deductible. Subrogation requests are handled by the Meridian claims desk.",
            "Lane exclusions: Carrier A does not serve TX-CA or AZ-CA lanes directly. For California destinations, Meridian routes via partner Carrier B with interline handoff in El Paso.",
            "Renewal terms: rate card refreshes annually on January 1. The current rate card is locked through 2026-12-31 with no mid-year adjustments.",
        ],
    },
    {
        "doc_type": "carrier_sla",
        "source": "carrier_agreements/carrier_b_2026.pdf",
        "metadata": {"carrier": "Carrier B", "effective_date": "2026-01-01"},
        "paragraphs": [
            "Carrier B SLA — Master Service Agreement effective 2026-01-01. Carrier B is a national LTL and FTL provider with broad coverage including all four Meridian priority lanes: TX-AZ, TX-TX, TX-CA, and AZ-CA.",
            "Surcharge structure: Carrier B applies a variable fuel surcharge indexed weekly to the DOE national diesel average. Recent history shows the surcharge ranging from 6% to 34% of linehaul on TX-AZ shipments, with a 90-day average of 19%.",
            "Accessorial fees: detention charged at $85/hour after 2 free hours. Lumper service at cost plus 15%. Reweigh fees of $35 per pallet apply to any shipment exceeding the booked weight by more than 3%.",
            "Transit time: Austin to Phoenix is quoted at 30-42 hours depending on driver rotation. On-time performance averaged 91.4% on TX-AZ in 2025.",
            "Capacity: no guaranteed weekly capacity on TX-AZ; bookings are spot-market. TX-CA has dedicated capacity of 8 truckloads/week with 7-day booking lead time.",
            "Historical risk note: Meridian internal review (2025 Q3) flagged Carrier B as carrying variable surcharge risk on intra-southwest lanes. One TX-AZ booking saw a 28% cost overrun versus quote due to fuel surcharge escalation between booking and delivery.",
            "Claims process: damage claims filed within 5 business days, $250 deductible per shipment, capped at $75,000 carrier liability. Claims beyond the cap require shipper-side cargo insurance.",
            "Equipment: 53' dry vans and 48' flatbeds. Reefer fleet limited to TX-CA lane only.",
        ],
    },
    {
        "doc_type": "carrier_sla",
        "source": "carrier_agreements/carrier_c_2026.pdf",
        "metadata": {"carrier": "Carrier C", "effective_date": "2026-01-01"},
        "paragraphs": [
            "Carrier C SLA — Master Service Agreement effective 2026-01-01. Carrier C specializes in expedited and high-value shipments across the western US, with strong coverage of AZ-CA and TX-CA lanes.",
            "Surcharge structure: Carrier C charges a fixed 7% premium over baseline linehaul to fund team-driver expedited service. No additional fuel surcharge is layered on top.",
            "Transit time: Phoenix to Los Angeles is 14-16 hours with team drivers. Austin to Los Angeles is 30-34 hours with a single driver hand-off in Tucson.",
            "Capacity: Carrier C operates 25 trucks dedicated to Meridian traffic, with daily departure slots on AZ-CA. TX-CA capacity is 6 trucks/week with 72-hour booking lead time.",
            "On-time performance: 96.8% on AZ-CA, 93.1% on TX-CA in 2025. Late deliveries beyond 2 hours trigger a 10% linehaul credit.",
            "Pricing: Carrier C is consistently the highest priced of the three primary Meridian carriers but offers the best transit time and reliability for high-value freight (>$50K cargo value).",
            "Equipment: 53' dry vans and reefers. All equipment is GPS-tracked with real-time temperature monitoring on reefer units, viewable in the Meridian shipper portal.",
        ],
    },
    # ---------------- Route Guides ----------------
    {
        "doc_type": "route_guide",
        "source": "route_guides/tx_az_lane.pdf",
        "metadata": {"lane": "TX-AZ", "effective_date": "2026-01-01"},
        "paragraphs": [
            "TX-AZ Route Guide. The TX-AZ lane covers shipments originating in Texas (Austin, Dallas, Houston, San Antonio, El Paso) and terminating in Arizona (Phoenix, Tucson, Flagstaff).",
            "Primary carrier: Carrier A is the preferred carrier for TX-AZ based on price, fixed surcharge structure, and dedicated weekly capacity. Use Carrier A as the default unless capacity is exhausted.",
            "Secondary carrier: Carrier B is the backup, but operations should be aware of the variable surcharge risk documented in the Carrier B SLA section 2.",
            "Standard transit: 36 hours door-to-door for Austin-Phoenix. Add 6 hours for Dallas origin or Tucson destination.",
            "Typical loaded weight range: 12,000-42,000 lbs. Shipments above 42,000 lbs require team-driver service — route via Carrier C and re-cost with the team premium.",
            "Booking lead time: 5 business days for Carrier A guaranteed capacity, 48 hours for Carrier A spot, 24 hours for Carrier B spot.",
            "Avoid: weekend departures from Austin between Memorial Day and Labor Day — historical detention rates spike due to driver shortages.",
        ],
    },
    {
        "doc_type": "route_guide",
        "source": "route_guides/tx_tx_lane.pdf",
        "metadata": {"lane": "TX-TX", "effective_date": "2026-01-01"},
        "paragraphs": [
            "TX-TX Route Guide. Intra-Texas shipments between any two of Austin, Dallas, Houston, San Antonio, El Paso, and Fort Worth.",
            "Primary carrier: Carrier A. Intra-Texas runs use Carrier A's dedicated dry-van fleet with no fuel surcharge applied on shipments under 20,000 lbs.",
            "Standard transit: same-day for Austin-Dallas-Houston triangle (under 250 miles), next-day for any pair involving El Paso.",
            "Equipment notes: most TX-TX moves are dry van; reefer available on request with 24 hours notice via Carrier A.",
            "Pricing: TX-TX is the lowest cost-per-mile lane in the Meridian network due to dense backhaul opportunities. Expect $1.85-$2.20 per loaded mile on standard dry van.",
            "Booking lead time: 24 hours for guaranteed Carrier A capacity. Same-day spot is possible Monday-Thursday.",
        ],
    },
    {
        "doc_type": "route_guide",
        "source": "route_guides/tx_ca_lane.pdf",
        "metadata": {"lane": "TX-CA", "effective_date": "2026-01-01"},
        "paragraphs": [
            "TX-CA Route Guide. Texas origins to California destinations (Los Angeles, San Diego, Oakland, Sacramento, Fresno).",
            "Primary carrier: Carrier C for expedited and high-value freight; Carrier B for standard freight under $25K cargo value.",
            "Carrier A does NOT serve TX-CA directly. Any TX-CA quote routed to Carrier A is rejected at the booking stage.",
            "Standard transit: 30-34 hours via Carrier C with single-driver handoff in Tucson. 48-60 hours via Carrier B.",
            "Pricing: Carrier B averages 22% lower than Carrier C on a per-mile basis, but the variable surcharge can erase that gap on volatile fuel weeks.",
            "California-specific compliance: CARB (California Air Resources Board) requires 2014-or-newer engines on all trucks entering CA. Both Carrier B and C fleets are CARB-compliant.",
        ],
    },
    {
        "doc_type": "route_guide",
        "source": "route_guides/az_ca_lane.pdf",
        "metadata": {"lane": "AZ-CA", "effective_date": "2026-01-01"},
        "paragraphs": [
            "AZ-CA Route Guide. Arizona origins (Phoenix, Tucson) to California destinations.",
            "Primary carrier: Carrier C. AZ-CA is Carrier C's strongest lane with daily departure slots from Phoenix.",
            "Secondary carrier: Carrier B for non-expedited freight. Booking lead time of 5 business days required.",
            "Standard transit: 14-16 hours Phoenix to Los Angeles with Carrier C team drivers. 28-32 hours with Carrier B single driver.",
            "Pricing: AZ-CA is the shortest of the three western lanes (~370 miles Phoenix-LA), so total cost differences between carriers are modest in absolute dollars.",
            "Avoid: I-10 westbound on Friday afternoons due to heavy Phoenix-LA passenger traffic causing 2-4 hour delays.",
        ],
    },
    # ---------------- Exception Playbooks ----------------
    {
        "doc_type": "exception_playbook",
        "source": "playbooks/late_delivery.pdf",
        "metadata": {"effective_date": "2026-01-01"},
        "paragraphs": [
            "Late Delivery Playbook. Triggered when a shipment misses its delivery appointment by more than 60 minutes without proactive carrier notification.",
            "Step 1: contact the carrier dispatch desk via the Meridian carrier portal and request a revised ETA. Document the new ETA in the shipment exception log.",
            "Step 2: notify the consignee by email with the revised ETA and an apology. Use the Meridian-standard late-delivery email template (T-LATE-01).",
            "Step 3: if the late delivery violates the carrier's on-time SLA (see the relevant carrier SLA section 3), file an automatic credit claim within 72 hours.",
            "Step 4: if the consignee requires goods by a hard deadline (e.g. production line, retail floor), evaluate expedited recovery via Carrier C team-driver service. Recovery shipments above $5K cost require shipper approval.",
            "Step 5: post-resolution, log root cause in the carrier scorecard. Three or more late deliveries on the same lane within a 30-day window trigger a carrier performance review.",
        ],
    },
    {
        "doc_type": "exception_playbook",
        "source": "playbooks/damaged_goods.pdf",
        "metadata": {"effective_date": "2026-01-01"},
        "paragraphs": [
            "Damaged Goods Playbook. Triggered when the consignee notes damage on the proof-of-delivery (POD) or files a damage claim within the carrier's claim window.",
            "Step 1: secure the damaged goods in place — do not return to shipper or dispose. Photograph the damage from at least four angles, including any visible carrier handling marks.",
            "Step 2: file the carrier damage claim within the carrier-specific window (Carrier A: 9 days, Carrier B: 5 days, Carrier C: 7 days). Include the photographs, POD, and bill of lading.",
            "Step 3: notify the Meridian claims desk via the claims-intake form. The claims desk will coordinate subrogation and any insurance claim above the carrier liability cap.",
            "Step 4: arrange replacement shipment if the consignee requires goods on the original timeline. Charge replacement to the open damage claim, not to the original PO.",
            "Step 5: log damage rate in carrier scorecard. Damage rate above 1.5% over a rolling 90-day window triggers a quality review.",
        ],
    },
    {
        "doc_type": "exception_playbook",
        "source": "playbooks/weight_discrepancy.pdf",
        "metadata": {"effective_date": "2026-01-01"},
        "paragraphs": [
            "Weight Discrepancy Playbook. Triggered when carrier-reported weight differs from booked weight by more than 3% or more than 500 lbs, whichever is greater.",
            "Step 1: pull the certified scale ticket from the carrier portal. The scale ticket is the authoritative weight for billing purposes.",
            "Step 2: if the discrepancy is on the high side, the carrier will apply a reweigh fee (Carrier B: $35/pallet). Approve the reweigh fee if the scale ticket is valid.",
            "Step 3: if the discrepancy is on the low side, request a billing correction from the carrier — booked weight should not increase the linehaul charge.",
            "Step 4: investigate the shipper-side cause. Common causes: incorrect product master weight in the WMS, missing dunnage in the weight calc, or partial loads booked as full.",
            "Step 5: update the product master weight in the WMS if a systemic error is found. Notify the operations lead so future bookings use the corrected weight.",
        ],
    },
    {
        "doc_type": "exception_playbook",
        "source": "playbooks/carrier_no_show.pdf",
        "metadata": {"effective_date": "2026-01-01"},
        "paragraphs": [
            "Carrier No-Show Playbook. Triggered when the assigned carrier fails to arrive at the pickup location within 2 hours of the scheduled pickup window.",
            "Step 1: contact the carrier dispatch desk immediately. Confirm whether the no-show is due to driver delay (recoverable) or equipment failure (not recoverable).",
            "Step 2: if not recoverable within 4 hours, escalate to spot-market re-sourcing. Use the Meridian spot quote tool to solicit bids from secondary carriers.",
            "Step 3: spot quotes above 15% over the original quote require operations manager approval. Quotes above $10K total require shipper approval per the shipping policy.",
            "Step 4: file a no-show penalty claim against the original carrier. Standard penalty is the higher of $500 or the spot-market markup, per the master carrier agreement.",
            "Step 5: log the no-show in the carrier scorecard. Two no-shows within a 60-day window trigger a capacity commitment review and possible volume reduction.",
        ],
    },
    # ---------------- Shipping Policies ----------------
    {
        "doc_type": "shipping_policy",
        "source": "policies/approval_thresholds.pdf",
        "metadata": {"effective_date": "2026-01-01"},
        "paragraphs": [
            "Shipping Approval Threshold Policy. This policy governs the dollar thresholds at which a shipping booking requires human approval before being confirmed with the carrier.",
            "Default threshold: any booking with an estimated total transportation cost greater than $10,000 USD requires explicit shipper approval via the Meridian shipper portal.",
            "The estimated total includes linehaul, fuel surcharge (using the carrier-specific structure — fixed for Carrier A, variable for Carrier B, no surcharge for Carrier C), accessorials, and any expedite premium.",
            "Bookings under $10,000 may be auto-confirmed by the agent if the recommended carrier matches the lane's primary carrier per the route guide.",
            "Tenant-level overrides: shippers may set a custom approval threshold via the Meridian admin console. The agent must always honor the tenant-level threshold over the default when one is recorded in long-term memory.",
            "Variable surcharge bookings: any quote from a carrier with a variable surcharge structure (e.g. Carrier B) that could plausibly escalate above the approval threshold within the surcharge volatility window must be flagged for approval even if the point estimate is under the threshold.",
        ],
    },
    {
        "doc_type": "shipping_policy",
        "source": "policies/vendor_list.pdf",
        "metadata": {"effective_date": "2026-01-01"},
        "paragraphs": [
            "Approved Vendor List Policy. Meridian maintains a curated approved-carrier list. Only carriers on the approved list may be booked through the agent.",
            "Current approved carriers: Carrier A, Carrier B, Carrier C. Any addition or removal requires a formal vendor onboarding review including insurance verification, safety rating check (FMCSA SMS), and a 30-day pilot run.",
            "Carrier substitution: if the primary carrier for a lane is unavailable, the agent may substitute with another approved carrier as documented in the relevant route guide. Off-list substitutions are prohibited.",
            "Insurance minimums: all approved carriers carry $1M auto liability, $100K cargo per shipment, and $1M general liability. Higher-value freight may require certificate-of-insurance verification before booking.",
            "Safety rating: approved carriers must hold a FMCSA Satisfactory or Conditional rating. Unsatisfactory rated carriers are immediately removed from the list pending review.",
        ],
    },
    # ---------------- General Freight Guidelines ----------------
    {
        "doc_type": "shipping_policy",
        "source": "policies/freight_guidelines_general.pdf",
        "metadata": {"effective_date": "2026-01-01"},
        "paragraphs": [
            "General Freight Guidelines. These guidelines apply to all Meridian-managed freight regardless of lane or carrier.",
            "Booking confirmations must include: origin and destination addresses, pickup and delivery appointment windows, commodity description, declared cargo value, total weight, total pallet count, and any special handling requirements.",
            "Pickup numbers and delivery numbers are required for all shipments to retail distribution centers. Missing reference numbers will be rejected at the receiving dock and re-delivered at shipper expense.",
            "Hazardous materials (hazmat) shipments require additional carrier qualification — none of Carrier A, B, or C are hazmat-certified. Hazmat moves are out-of-scope for this agent.",
            "Photographs of the loaded trailer are required on departure for any shipment with declared cargo value above $50K. Photos are uploaded to the Meridian shipper portal.",
        ],
    },
    {
        "doc_type": "shipping_policy",
        "source": "policies/freight_guidelines_appointments.pdf",
        "metadata": {"effective_date": "2026-01-01"},
        "paragraphs": [
            "Appointment Window Guidelines. All deliveries to Meridian-managed customer locations require a confirmed delivery appointment.",
            "Standard appointment window is 2 hours. Tight appointments (under 1 hour) require carrier acceptance at booking time and may carry an expedite premium.",
            "Same-day appointment changes require operations lead approval. Appointment changes are charged a $75 administrative fee plus any carrier-side rescheduling cost.",
            "Detention starts after 2 free hours at the dock for all approved carriers. Detention rates: Carrier A $65/hour, Carrier B $85/hour, Carrier C $95/hour.",
            "Drop-trailer programs: Carrier A offers drop-trailer service on TX-TX with a $250 weekly trailer rental. Drop-trailer eliminates detention exposure for high-volume facilities.",
        ],
    },
    {
        "doc_type": "shipping_policy",
        "source": "policies/claims_and_insurance.pdf",
        "metadata": {"effective_date": "2026-01-01"},
        "paragraphs": [
            "Claims and Insurance Policy. This policy covers cargo claims, carrier liability limits, and shipper-side insurance requirements.",
            "Carrier liability: Carrier A self-insures to $100K per shipment with no deductible. Carrier B caps at $75K with a $250 deductible. Carrier C caps at $100K with a $500 deductible.",
            "Shipper-side cargo insurance is required for any shipment with declared value above the relevant carrier liability cap. The Meridian procurement team maintains a master policy that covers excess up to $5M per shipment.",
            "Claim filing windows are carrier-specific. Missing the window forfeits the right to recover from the carrier. The agent should always cite the applicable window when discussing claims.",
            "Subrogation: Meridian claims desk handles all subrogation against carriers above $10K in claimed value. Below $10K, shippers may file directly via the carrier portal.",
        ],
    },
    {
        "doc_type": "shipping_policy",
        "source": "policies/hours_of_service.pdf",
        "metadata": {"effective_date": "2026-01-01"},
        "paragraphs": [
            "Hours of Service Guidelines. US DOT hours-of-service (HOS) regulations limit single drivers to 11 hours of driving in a 14-hour duty period, followed by 10 consecutive hours off-duty.",
            "Single-driver runs above 600 miles typically require an overnight rest break and should be quoted with a 1.5-day transit minimum. Team-driver service avoids the rest break by alternating drivers.",
            "Carrier A and Carrier B operate single-driver fleets on the TX-AZ and TX-TX lanes. Team-driver service is available on TX-AZ via Carrier A with 72 hours notice at a $0.22/mile premium.",
            "Carrier C operates team drivers as the default on all AZ-CA and TX-CA shipments, which is why their transit times are materially faster on those lanes.",
            "Detention time counts against the driver's 14-hour duty clock. Excessive detention at a shipper facility can force the driver to take a 10-hour break before continuing, adding a full calendar day to transit.",
        ],
    },
    # ---------------- Fuel Surcharge Methodology ----------------
    {
        "doc_type": "shipping_policy",
        "source": "policies/fuel_surcharge_methodology.pdf",
        "metadata": {"effective_date": "2026-01-01"},
        "paragraphs": [
            "Fuel Surcharge Methodology. This document defines how the agent should reason about fuel surcharges when comparing carrier quotes on the same lane.",
            "Carrier A applies a per-lane weight-threshold model: shipments at or under the threshold incur zero fuel surcharge, shipments over the threshold incur a flat 4.5% on linehaul. Thresholds are 22,000 lbs on TX-AZ and 20,000 lbs on TX-TX. This makes Carrier A's total cost deterministic at quote time.",
            "Carrier B applies a DOE-indexed variable surcharge with no weight threshold (the 15,000 lb value in the SLA refers to the minimum chargeable weight, not a surcharge floor). The published rate of 8.5% on TX-AZ is a 90-day average; week-over-week the surcharge has ranged from 6% to 34%. The agent must always disclose this volatility when quoting Carrier B.",
            "Carrier C bundles a fixed 7% expedite premium into linehaul on AZ-CA and TX-CA and applies a 5-12% variable surcharge on TX-AZ and TX-NM. Carrier C is rarely surcharge-optimal but is the only viable choice when transit time is the binding constraint.",
            "Decision rule: when a shipper question filters on 'no fuel surcharge', only SLAs with `surcharge_rate == 0.0` qualify. On TX-AZ, the only SLA meeting that filter is Carrier A's (`sla_a_txaz`).",
            "Decision rule: when a shipper question filters on 'weight threshold above N lbs', compare against `weight_threshold_lb` from the relevant SLA row. On TX-AZ at N=18,000 lbs, only Carrier A qualifies (threshold 22,000 lbs); Carrier B (15,000) and Carrier C (10,000) do not.",
            "When both filters are combined ('no surcharge AND threshold > 18,000 lbs'), Carrier A is the unique answer on TX-AZ. Cite this document plus `carrier_agreements/carrier_a_2026.pdf` in the response.",
        ],
    },
    # ---------------- Customer Tenant Profile ----------------
    {
        "doc_type": "shipping_policy",
        "source": "policies/customer_profile_acme.pdf",
        "metadata": {"customer": "Acme Manufacturing", "effective_date": "2026-01-01"},
        "paragraphs": [
            "Acme Manufacturing — Tenant Profile. Acme is a Texas-based industrial parts manufacturer with primary distribution centers in Austin and Dallas, and a secondary depot in Phoenix.",
            "Lane mix (2025 actuals): 62% TX-TX (Austin-Dallas-Houston triangle), 24% TX-AZ (Austin/Dallas to Phoenix/Tucson), 9% TX-NM, 5% other. The agent should default routing optimizations to the TX-TX and TX-AZ playbooks first.",
            "Approval threshold: Acme has set a tenant-level threshold of $10,000 USD per booking. Any quote above this threshold must be routed for human approval via the Meridian shipper portal — overriding the network default only if Acme raises it in writing.",
            "Preferred carrier: Carrier A on TX-AZ and TX-TX, per Acme's 2026 carrier qualification review. Carrier B is approved as backup only; the agent should append the variable-surcharge warning whenever Carrier B is recommended on TX-AZ (per procedural rule proc_002).",
            "Service level expectations: Acme requires same-day delivery on the Austin-Dallas hot lane Monday-Thursday and next-day Friday. Phoenix deliveries are 48-hour standard, with expedited 24-hour service permitted up to $14,500 per booking with approval.",
            "Billing: Acme is on net-30 terms with consolidated weekly invoicing. The agent does not need to surface payment terms in routine quotes.",
            "Escalation contacts: Operations lead is Maria Chen (logistics@acme.example), approver of record for the >$10K threshold. CFO sign-off required above $50K.",
        ],
    },
    # ---------------- Austin-Dallas Hot-Lane Guide ----------------
    {
        "doc_type": "route_guide",
        "source": "route_guides/austin_dallas_hot_lane.pdf",
        "metadata": {"lane": "TX-TX", "sub_lane": "AUS-DAL", "effective_date": "2026-01-01"},
        "paragraphs": [
            "Austin-Dallas Hot Lane. The AUS-DAL sub-lane (~195 miles via I-35) is the highest-volume corridor in the Meridian-managed Texas network and is operated as a same-day turn Monday-Thursday.",
            "Primary carrier: Carrier A dry van with no fuel surcharge under the 20,000 lb TX-TX threshold. Typical 15,000 lb dry-van quotes price between $410 and $475 all-in.",
            "Standard transit: 4-5 hours driving time. With one 30-minute mandatory HOS break, a single driver dispatched at 06:00 can complete pickup, transit, and delivery before the 14-hour duty period closes at 20:00 — making same-day turn comfortably feasible.",
            "Hours of service: Austin-Dallas single-driver runs are well within the 11-hour driving limit. Detention at the dock is the only typical risk to same-day completion; if a driver is held more than 3 hours at the shipper, the run rolls to next-day.",
            "Booking lead time: 24 hours for guaranteed Carrier A capacity. Same-day spot booking is possible before 10:00 Monday-Thursday subject to fleet availability.",
            "Drop-trailer program: Acme's Austin DC participates in Carrier A's drop-trailer program ($250/week trailer rental), which eliminates detention exposure and supports a 23:00 dispatch / 06:00 delivery overnight option.",
            "Friday and weekend service: Friday dispatch must be confirmed by 14:00 to guarantee same-day. Saturday/Sunday runs are spot-only and incur a 12% weekend premium.",
        ],
    },
    # ---------------- Carrier A Expedited Addendum ----------------
    {
        "doc_type": "carrier_sla",
        "source": "carrier_agreements/carrier_a_expedited_addendum_2026.pdf",
        "metadata": {"carrier": "Carrier A", "effective_date": "2026-02-15"},
        "paragraphs": [
            "Carrier A Expedited Service Addendum — effective 2026-02-15. This addendum extends the Carrier A master agreement to cover high-weight and expedited bookings on the TX-AZ lane.",
            "Heavy-haul tier: shipments between 40,000 and 45,000 lbs on TX-AZ are quoted under the heavy-haul tier with a 22% premium over standard linehaul. Booking lead time is 72 hours. Equipment is reinforced 53' dry van with 45,000 lb cargo capacity.",
            "Expedited team-driver service: Austin-Phoenix can be quoted as a 24-hour team-driver run at $0.22/mile premium over standard. Typical 42,000 lb expedited Austin-Phoenix quotes price between $13,800 and $14,900 all-in.",
            "Heavy-haul + expedited combined: any quote combining heavy-haul tier (40,000+ lbs) and expedited transit always exceeds the $10,000 approval threshold and must be routed for human approval. The agent should never auto-confirm a booking in this combined category.",
            "Capacity: 2 dedicated heavy-haul expedited slots per week on TX-AZ. Slots are first-come-first-served and not reservable beyond the 72-hour lead time.",
            "Surcharge: the standard 4.5% above-threshold fuel surcharge does not apply to expedited heavy-haul; the 22% premium and team-driver $/mile premium are inclusive of fuel cost recovery.",
            "Performance: 96.5% on-time on expedited bookings in 2025. Late deliveries beyond 2 hours past the appointment trigger a 10% linehaul credit, double the standard tier credit.",
        ],
    },
    # ---------------- Carrier Scorecards ----------------
    {
        "doc_type": "carrier_scorecard",
        "source": "scorecards/2026_q1_carrier_summary.pdf",
        "metadata": {"period": "2026-Q1", "effective_date": "2026-04-15"},
        "paragraphs": [
            "Q1 2026 Carrier Scorecard Summary. This report consolidates on-time performance, damage rates, and cost variance for the three approved Meridian carriers across the period 2026-01-01 through 2026-03-31.",
            "Carrier A — TX-AZ: 98.4% on-time (target 98.0%), 0.4% damage rate (target <1.5%), $0 cost variance vs. quote. Recommended for continued primary-lane status on TX-AZ and TX-TX.",
            "Carrier A — TX-TX: 99.1% on-time, 0.2% damage rate, $0 cost variance. Best-in-network performance on Austin-Dallas same-day turns.",
            "Carrier B — TX-AZ: 91.2% on-time (below 95% target), 0.9% damage rate, +18% average cost variance vs. quote driven by fuel surcharge volatility. Performance review flagged — Acme should continue to limit Carrier B to backup status on TX-AZ.",
            "Carrier B — TX-NM: 94.8% on-time, 1.1% damage rate, +6% cost variance. Acceptable for primary-lane status on TX-NM where surcharge has been less volatile.",
            "Carrier C — TX-AZ: 95.7% on-time, 0.3% damage rate, $0 cost variance (fixed expedite premium). Justifies third-tier ranking on TX-AZ for time-critical freight only.",
            "Recommendation: no changes to the approved vendor list this quarter. Acme's preference for Carrier A on TX-AZ and TX-TX is reinforced by Q1 metrics. Re-evaluate Carrier B's TX-AZ status if Q2 surcharge volatility persists.",
        ],
    },
]
