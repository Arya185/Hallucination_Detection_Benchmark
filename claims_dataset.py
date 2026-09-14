import json
import os
from collections import defaultdict
from tqdm import tqdm
from openai import OpenAI

dataset = []

# ===================== DOC1: NASA Chandra hypersoft X-ray sources =====================
doc1_text = """NASA Chandra discovers “hypersoft” X-ray sources
On September 9, 2026, NASA’s Chandra X-ray Observatory reported the discovery of a previously unrecognized class of astronomical objects called hypersoft X-ray sources. The objects were identified because they appear strongly at very low X-ray energies but disappear from images at higher X-ray energies. The researchers also found that these sources produce unusually intense ultraviolet radiation.
The team searched publicly available Chandra archival observations covering six galaxies and identified 84 hypersoft X-ray sources. Two of the galaxies are spiral galaxies, M31 and M101, while the remaining four are elliptical galaxies. The sources occur both in regions containing active star formation and in regions dominated by older stellar populations. (Chandra X-ray Observatory)
The physical nature of the objects remains uncertain. Researchers suggest that they could involve compact objects such as black holes, neutron stars, or white dwarfs drawing material from companion stars. In such systems, material transferred from a companion can become extremely hot and emit X-rays.
The discovery may have implications beyond identifying a new type of X-ray source. Researchers are interested in whether some of these systems could be related to the progenitors of Type Ia supernovae, which are important astronomical distance indicators and played a major role in establishing that the expansion of the universe is accelerating. The sources may also contribute ultraviolet radiation capable of stripping electrons from gas between stars in some galaxies. (Chandra X-ray Observatory)
The objects were difficult to identify partly because their low-energy X-rays are challenging for X-ray observatories to detect. Their ultraviolet radiation is also readily absorbed by hydrogen and helium gas between stars. By searching archival data specifically for their unusual spectral behavior, the researchers were able to identify the population.
The researchers emphasize that the nature and eventual evolution of the objects are still uncertain. The discovery therefore represents a starting point for further investigation rather than a definitive explanation of their physical origin. (Chandra X-ray Observatory)"""

doc1_claims = [
    {"claim_id": "doc1_c01", "claim_text": "Researchers identified 84 hypersoft X-ray sources across six galaxies.", "category": "numerical", "is_unsupported": False},
    {"claim_id": "doc1_c02", "claim_text": "The observations covered 12 galaxies in total.", "category": "numerical", "is_unsupported": True},
    {"claim_id": "doc1_c03", "claim_text": "The discovery was made using NASA's Chandra X-ray Observatory.", "category": "entity", "is_unsupported": False},
    {"claim_id": "doc1_c04", "claim_text": "The lead researcher was Mustafa Muhibullah of the University of Alabama.", "category": "entity", "is_unsupported": False},
    {"claim_id": "doc1_c05", "claim_text": "The hypersoft sources have been proven to be Type Ia supernova progenitors.", "category": "causal", "is_unsupported": True},
    {"claim_id": "doc1_c06", "claim_text": "Their ultraviolet radiation may contribute to ionizing interstellar gas in some galaxies.", "category": "causal", "is_unsupported": False},
    {"claim_id": "doc1_c07", "claim_text": "The sources could involve compact objects such as black holes, neutron stars, or white dwarfs.", "category": "hedged_plausible", "is_unsupported": False},
    {"claim_id": "doc1_c08", "claim_text": "Some of these systems might eventually help researchers understand Type Ia supernova progenitors.", "category": "hedged_plausible", "is_unsupported": False},
    {"claim_id": "doc1_c09", "claim_text": "M31 and M101 were among the six galaxies searched.", "category": "supported_control", "is_unsupported": False},
    {"claim_id": "doc1_c10", "claim_text": "The researchers used publicly available Chandra archival data.", "category": "supported_control", "is_unsupported": False},
]

for c in doc1_claims:
    dataset.append({
        "claim_id": c["claim_id"],
        "doc_id": "doc1",
        "source_text": doc1_text,
        "claim_text": c["claim_text"],
        "category": c["category"],
        "is_unsupported": c["is_unsupported"]
    })

# ===================== DOC2: HPE Q3 FY2026 =====================
doc2_text = """Hewlett Packard Enterprise reports Q3 FY2026 results
Hewlett Packard Enterprise (HPE) announced its fiscal 2026 third-quarter results on September 2, 2026, reporting record revenue and profitability. The quarter ended July 31, 2026. HPE reported $12.2 billion in revenue, representing a 34% increase from the same period a year earlier. GAAP operating profit increased 464% year over year, while non-GAAP operating profit increased 155%. (Hewlett Packard Enterprise)
HPE's Cloud & AI segment generated $9.0 billion in revenue, up 25.4% year over year, and recorded a 17.0% operating profit margin. Within that segment, server revenue reached $6.8 billion, while storage generated $1.3 billion and financial services generated $0.9 billion.
Networking revenue was $2.9 billion, up 74.9%. Data Center Networking revenue increased 112.2% to $382 million, while routing revenue rose 270.0% to $788 million. The company described customer demand and its order backlog as important contributors to its results. (Hewlett Packard Enterprise)
HPE also reported $1.6 billion in cash flow from operations and $1.0 billion in free cash flow. The company returned $324 million to common shareholders through dividends and share repurchases.
The company declared a regular cash dividend of $0.1425 per share, payable on or about October 16, 2026, to shareholders of record as of September 17. HPE also raised its fourth-quarter expectations, forecasting revenue between $13.9 billion and $14.8 billion and GAAP diluted EPS between $1.12 and $1.22. (Hewlett Packard Enterprise)
HPE management described AI as a multi-year growth opportunity, but the earnings release does not establish that AI alone caused the company's overall revenue growth. The results cover several business segments, including networking, servers, storage, and financial services.
The release therefore provides concrete financial results as well as management's interpretation of the company's performance. Statements about future growth remain forecasts rather than realized financial outcomes."""

doc2_claims = [
    {"claim_id": "doc2_c01", "claim_text": "HPE reported $12.2 billion in Q3 FY2026 revenue.", "category": "numerical", "is_unsupported": False},
    {"claim_id": "doc2_c02", "claim_text": "HPE's Networking revenue increased by 74.9% year over year.", "category": "numerical", "is_unsupported": False},
    {"claim_id": "doc2_c03", "claim_text": "Antonio Neri was HPE's president and CEO.", "category": "entity", "is_unsupported": False},
    {"claim_id": "doc2_c04", "claim_text": "Marie Myers was HPE's executive vice president and CFO.", "category": "entity", "is_unsupported": False},
    {"claim_id": "doc2_c05", "claim_text": "HPE's 34% revenue growth was entirely caused by AI demand.", "category": "causal", "is_unsupported": True},
    {"claim_id": "doc2_c06", "claim_text": "The increase in Data Center Networking revenue caused HPE's overall operating profit to rise 464%.", "category": "causal", "is_unsupported": True},
    {"claim_id": "doc2_c07", "claim_text": "AI may become a significant multi-year growth driver for HPE.", "category": "hedged_plausible", "is_unsupported": False},
    {"claim_id": "doc2_c08", "claim_text": "HPE's record order backlog could support stronger future revenue.", "category": "hedged_plausible", "is_unsupported": False},
    {"claim_id": "doc2_c09", "claim_text": "HPE forecast Q4 FY2026 revenue of $13.9–$14.8 billion.", "category": "supported_control", "is_unsupported": False},
    {"claim_id": "doc2_c10", "claim_text": "HPE declared a quarterly dividend of $0.1425 per share.", "category": "supported_control", "is_unsupported": False},
]

for c in doc2_claims:
    dataset.append({
        "claim_id": c["claim_id"],
        "doc_id": "doc2",
        "source_text": doc2_text,
        "claim_text": c["claim_text"],
        "category": c["category"],
        "is_unsupported": c["is_unsupported"]
    })

# ===================== DOC3: NYC Executive Order No. 22 =====================
doc3_text = """New York City establishes a citywide continuity framework
On September 4, 2026, New York City issued Executive Order No. 22, establishing a new framework for maintaining essential city services during emergencies and other disruptions. The order covers services including public safety, health, transportation, water, education, social services, justice administration, and communication of critical information. (New York City Government)
The order creates a Continuity and Mission Assurance Team (CMAT) as the city's senior interagency body for continuity-of-operations, continuity-of-government, and mission-assurance policy. The CMAT is chaired by the Deputy Mayor for Operations or the Deputy Mayor's designee.
Its core membership includes senior representatives from New York City Emergency Management (NYCEM), the Department of Citywide Administrative Services (DCAS), the Office of Management and Budget (OMB), and the Office of Technology and Innovation (OTI). The order permits additional agencies or offices to participate when determined by the chair. (New York City Government)
NYCEM is designated as the lead coordinating agency for the city's continuity and mission-assurance programs. Among other responsibilities, NYCEM is tasked with developing and maintaining citywide continuity policies, administering enterprise continuity-planning software, and supporting interagency readiness activities.
The order requires city agencies to maintain Continuity of Operations (COOP) plans addressing essential services and their role in sustaining city government. These plans must be updated annually, and agencies must conduct at least one exercise each year to test their plans. Agencies must also designate a senior official responsible for continuity planning and readiness. (New York City Government)
The order establishes a framework for identifying Citywide Priority Essential Services, which are services whose disruption could produce significant cascading effects on New Yorkers or city government.
Executive Order No. 22 also revokes Executive Order No. 107, issued on October 2, 2007. The new order took effect immediately.
The policy is primarily an organizational and preparedness framework. It does not state that all city services will literally continue without interruption under every emergency; rather, it requires planning intended to maintain or rapidly resume essential services.
The CMAT must meet regularly, at least quarterly, with additional meetings when circumstances require."""

doc3_claims = [
    {"claim_id": "doc3_c01", "claim_text": "Executive Order No. 22 was issued on September 4, 2026.", "category": "numerical", "is_unsupported": False},
    {"claim_id": "doc3_c02", "claim_text": "Agencies are required to conduct at least one COOP exercise annually.", "category": "numerical", "is_unsupported": False},
    {"claim_id": "doc3_c03", "claim_text": "NYC Emergency Management (NYCEM) is designated as the lead coordinating agency.", "category": "entity", "is_unsupported": False},
    {"claim_id": "doc3_c04", "claim_text": "Office of Technology and Innovation (OTI) is part of the CMAT's core representation.", "category": "entity", "is_unsupported": False},
    {"claim_id": "doc3_c05", "claim_text": "The order guarantees uninterrupted delivery of every city service during emergencies.", "category": "causal", "is_unsupported": True},
    {"claim_id": "doc3_c06", "claim_text": "Annual COOP exercises will prevent major disruptions to essential services.", "category": "causal", "is_unsupported": True},
    {"claim_id": "doc3_c07", "claim_text": "The framework could improve citywide coordination during disruptions.", "category": "hedged_plausible", "is_unsupported": False},
    {"claim_id": "doc3_c08", "claim_text": "Regular exercises may reveal weaknesses in agency continuity plans.", "category": "hedged_plausible", "is_unsupported": False},
    {"claim_id": "doc3_c09", "claim_text": "Executive Order No. 107 from 2007 was revoked.", "category": "supported_control", "is_unsupported": False},
    {"claim_id": "doc3_c10", "claim_text": "The CMAT is chaired by the Deputy Mayor for Operations or their designee.", "category": "supported_control", "is_unsupported": False},
    {"claim_id": "doc3_c11", "claim_text": "CMAT must meet at least quarterly.", "category": "numerical", "is_unsupported": False},
    {"claim_id": "doc3_c12", "claim_text": "The new order took effect immediately.", "category": "supported_control", "is_unsupported": False},
]

for c in doc3_claims:
    dataset.append({
        "claim_id": c["claim_id"],
        "doc_id": "doc3",
        "source_text": doc3_text,
        "claim_text": c["claim_text"],
        "category": c["category"],
        "is_unsupported": c["is_unsupported"]
    })

# ===================== DOC4: PrevenTRON =====================
doc4_text = """PrevenTRON begins Alzheimer’s prevention research
An international Phase III clinical trial called PrevenTRON is investigating whether an experimental drug called trontinemab can delay or prevent Alzheimer's disease before symptoms appear. The study is recruiting adults considered to be at high risk of developing Alzheimer's but who currently have no memory or thinking problems. (Surrey and Borders Partnership)
The trial plans to recruit approximately 1,600 adults aged 55–80 at research sites around the world. Surrey and Borders Partnership NHS Foundation Trust is the lead UK recruitment site and says it can recruit eligible participants from anywhere in the United Kingdom. The organization describes itself as the first site in the UK and Europe and the third site globally to offer the study. (Surrey and Borders Partnership)
Trontinemab is being investigated for its ability to remove amyloid plaques that accumulate in the brain and are associated with Alzheimer's disease. Screening includes blood testing to identify people who may be appropriate for participation.
Importantly, PrevenTRON is an investigational study. The trial is designed to determine whether administering the drug before symptoms appear can actually delay or prevent the development of Alzheimer's disease. The existence of the trial does not establish that trontinemab can prevent Alzheimer's.
The study therefore represents a shift toward investigating interventions at an earlier stage of disease risk. Instead of recruiting people who already have cognitive symptoms, the researchers are studying individuals who are currently asymptomatic but considered at elevated risk.
The trial is sponsored by Roche, the pharmaceutical company developing trontinemab. The NHS recruitment page emphasizes that participants must pass a screening process before being accepted into the study. (Surrey and Borders Partnership)
The study's large planned enrollment is intended to provide evidence about whether the intervention has a meaningful effect on the onset or progression of Alzheimer's disease. However, the eventual efficacy and safety findings cannot be inferred from the trial's launch announcement. Those questions require analysis of the trial's results after participants have been followed."""

doc4_claims = [
    {"claim_id": "doc4_c01", "claim_text": "PrevenTRON plans to recruit approximately 1,600 participants.", "category": "numerical", "is_unsupported": False},
    {"claim_id": "doc4_c02", "claim_text": "Eligible participants are 55–80 years old.", "category": "numerical", "is_unsupported": False},
    {"claim_id": "doc4_c03", "claim_text": "Trontinemab is the investigational drug being studied.", "category": "entity", "is_unsupported": False},
    {"claim_id": "doc4_c04", "claim_text": "Roche sponsors the PrevenTRON study.", "category": "entity", "is_unsupported": False},
    {"claim_id": "doc4_c05", "claim_text": "Trontinemab has been proven to prevent Alzheimer's disease before symptoms appear.", "category": "causal", "is_unsupported": True},
    {"claim_id": "doc4_c06", "claim_text": "Removing amyloid plaques necessarily prevents the development of Alzheimer's disease.", "category": "causal", "is_unsupported": True},
    {"claim_id": "doc4_c07", "claim_text": "Trontinemab could potentially delay Alzheimer's onset if the trial demonstrates efficacy.", "category": "hedged_plausible", "is_unsupported": False},
    {"claim_id": "doc4_c08", "claim_text": "Earlier intervention may offer a useful strategy for Alzheimer's prevention research.", "category": "hedged_plausible", "is_unsupported": False},
    {"claim_id": "doc4_c09", "claim_text": "PrevenTRON is described as a Phase III research study.", "category": "supported_control", "is_unsupported": False},
    {"claim_id": "doc4_c10", "claim_text": "The study targets people without current Alzheimer's symptoms or memory problems.", "category": "supported_control", "is_unsupported": False},
]

for c in doc4_claims:
    dataset.append({
        "claim_id": c["claim_id"],
        "doc_id": "doc4",
        "source_text": doc4_text,
        "claim_text": c["claim_text"],
        "category": c["category"],
        "is_unsupported": c["is_unsupported"]
    })

# ===================== DOC5: Atomic hydrogen =====================
doc5_text = """Weak evolution of cosmic atomic hydrogen
A study published in Nature Astronomy in September 2026 examined how the amount of atomic hydrogen in galaxies has changed over cosmic time. Researchers used spectroscopy from more than 2.5 million galaxies to investigate the relationship between atomic hydrogen and the decline in star formation over approximately the last 4.5 billion years.
The study reports that the Universe appears to have retained most of its atomic hydrogen during this period, despite a substantial decline in the rate at which new stars formed. This result challenges a simple interpretation in which declining star formation is primarily explained by galaxies simply running out of their atomic-gas fuel.
The researchers argue that the results instead point toward inefficient conversion of atomic hydrogen into stars as an important factor. Atomic hydrogen can serve as a reservoir from which molecular gas forms, and molecular gas is more directly associated with star formation.
The work relies on a very large spectroscopic galaxy sample, allowing researchers to investigate population-level trends rather than focusing on individual galaxies. The analysis provides evidence concerning the evolution of the cosmic gas reservoir, but it does not mean that every galaxy has retained exactly the same amount of hydrogen.
The finding is particularly relevant because the decline of cosmic star formation remains an important question in galaxy evolution. If atomic hydrogen has not been substantially depleted, researchers need to understand why galaxies have become less efficient at turning available gas into stars.
The result also does not establish a single mechanism responsible for the entire decline in star formation. Processes such as feedback from stars and active galactic nuclei, gas heating, environmental effects, and changes in the conversion between atomic and molecular gas can all influence galaxy evolution.
The study therefore provides a population-scale constraint on models of galaxy evolution rather than a definitive explanation for the history of star formation.
(Nature)"""

doc5_claims = [
    {"claim_id": "doc5_c01", "claim_text": "The study analyzed spectroscopy from more than 2.5 million galaxies.", "category": "numerical", "is_unsupported": False},
    {"claim_id": "doc5_c02", "claim_text": "It examined approximately 4.5 billion years of cosmic history.", "category": "numerical", "is_unsupported": False},
    {"claim_id": "doc5_c03", "claim_text": "The study was published in Nature Astronomy.", "category": "entity", "is_unsupported": False},
    {"claim_id": "doc5_c04", "claim_text": "The research concerns the evolution of atomic hydrogen in galaxies.", "category": "entity", "is_unsupported": False},
    {"claim_id": "doc5_c05", "claim_text": "The study proves that declining star formation is caused exclusively by inefficient hydrogen conversion.", "category": "causal", "is_unsupported": True},
    {"claim_id": "doc5_c06", "claim_text": "Retaining atomic hydrogen directly prevents galaxies from forming new stars.", "category": "causal", "is_unsupported": True},
    {"claim_id": "doc5_c07", "claim_text": "Inefficient conversion of atomic gas into star-forming material may help explain declining star formation.", "category": "hedged_plausible", "is_unsupported": False},
    {"claim_id": "doc5_c08", "claim_text": "Feedback processes could contribute to the observed decline in star formation.", "category": "hedged_plausible", "is_unsupported": False},
    {"claim_id": "doc5_c09", "claim_text": "The study reports that the Universe retained much of its atomic hydrogen over the period examined.", "category": "supported_control", "is_unsupported": False},
    {"claim_id": "doc5_c10", "claim_text": "The study examines a population-level trend rather than one individual galaxy.", "category": "supported_control", "is_unsupported": False},
]

for c in doc5_claims:
    dataset.append({
        "claim_id": c["claim_id"],
        "doc_id": "doc5",
        "source_text": doc5_text,
        "claim_text": c["claim_text"],
        "category": c["category"],
        "is_unsupported": c["is_unsupported"]
    })

# ===================== DOC6: HCLTech =====================
doc6_text = """HCLTech launches semiconductor engineering laboratory
HCLTech announced on September 8, 2026, that it had launched an Advanced Semiconductor Lab in Bengaluru, India. The company says the facility is intended to expand its semiconductor engineering and testing capabilities and support global clients.
HCLTech announced an investment of ₹185 crore in the facility. The laboratory covers approximately 40,000 square feet, including 25,000 square feet of Class 10K and Class 1K cleanrooms.
The laboratory is designed to support post-silicon engineering, advanced semiconductor testing, and failure analysis. HCLTech describes the facility as an end-to-end post-silicon engineering and testing environment rather than simply a conventional software-development center.
The launch is part of HCLTech's broader semiconductor strategy. The company says it has approximately two decades of experience in semiconductor engineering and intends to use the new facility to accelerate development and testing for customers.
The facility's cleanroom infrastructure is intended to support controlled semiconductor engineering environments. Class 10K and Class 1K classifications refer to cleanliness levels based on allowable airborne particle concentrations.
The announcement does not mean that HCLTech has begun manufacturing advanced semiconductor chips at the Bengaluru site. The stated focus is on engineering, testing, and failure analysis after silicon has been produced.
The company also describes the investment as contributing to India's semiconductor ecosystem. Such a claim concerns the facility's potential role in the broader ecosystem; it does not by itself establish that the facility will materially change India's global semiconductor market share.
The announcement therefore combines concrete facility specifications with broader strategic claims about semiconductor innovation and India's position in the global chip ecosystem.
(HCLTech)"""

doc6_claims = [
    {"claim_id": "doc6_c01", "claim_text": "HCLTech invested ₹185 crore in the new laboratory.", "category": "numerical", "is_unsupported": False},
    {"claim_id": "doc6_c02", "claim_text": "The laboratory covers 40,000 square feet.", "category": "numerical", "is_unsupported": False},
    {"claim_id": "doc6_c03", "claim_text": "The facility is located in Bengaluru.", "category": "entity", "is_unsupported": False},
    {"claim_id": "doc6_c04", "claim_text": "HCLTech launched the facility as its Advanced Semiconductor Lab.", "category": "entity", "is_unsupported": False},
    {"claim_id": "doc6_c05", "claim_text": "The laboratory will make India the world's largest semiconductor manufacturing country.", "category": "causal", "is_unsupported": True},
    {"claim_id": "doc6_c06", "claim_text": "The ₹185 crore investment will eliminate India's semiconductor testing shortage.", "category": "causal", "is_unsupported": True},
    {"claim_id": "doc6_c07", "claim_text": "The laboratory could strengthen HCLTech's semiconductor engineering capabilities.", "category": "hedged_plausible", "is_unsupported": False},
    {"claim_id": "doc6_c08", "claim_text": "The facility may help shorten development and testing cycles for some customers.", "category": "hedged_plausible", "is_unsupported": False},
    {"claim_id": "doc6_c09", "claim_text": "The facility includes Class 10K and Class 1K cleanrooms.", "category": "supported_control", "is_unsupported": False},
    {"claim_id": "doc6_c10", "claim_text": "Its stated capabilities include post-silicon engineering, advanced testing and failure analysis.", "category": "supported_control", "is_unsupported": False},
]

for c in doc6_claims:
    dataset.append({
        "claim_id": c["claim_id"],
        "doc_id": "doc6",
        "source_text": doc6_text,
        "claim_text": c["claim_text"],
        "category": c["category"],
        "is_unsupported": c["is_unsupported"]
    })

# ===================== DOC7: AbbVie CERVINO =====================
doc7_text = """AbbVie reports Phase 3 CERVINO results
AbbVie announced positive topline results from the Phase 3 CERVINO trial on September 3, 2026. The study evaluated etentamig, an investigational BCMA × CD3 bispecific T-cell engager, in patients with triple-class-exposed relapsed or refractory multiple myeloma.
The trial compared etentamig with investigator-selected standard available therapies. According to AbbVie, the study met its dual primary endpoints of objective response rate (ORR) and progression-free survival (PFS).
At the data cutoff, the trial included 393 patients, who had received a median of three previous lines of therapy. Median follow-up was 11.4 months.
AbbVie reported an objective response rate of 74% for etentamig and a hazard ratio of 0.40 for progression-free survival. The company characterized this as a 60% reduction in the relative risk of disease progression or death compared with the control group.
The company also reported low rates of cytokine release syndrome and fatal infections with the studied monthly dosing schedule following a single step-up dose.
The results are described as topline results, meaning that the announcement does not provide the full dataset. AbbVie said the complete results would be presented at the International Myeloma Society Annual Meeting in September.
The drug remains investigational. Positive Phase 3 topline findings do not automatically mean that the treatment is approved or that it will become the preferred therapy for multiple myeloma.
AbbVie said it planned to discuss the results with global regulatory authorities to determine next steps.
The announcement therefore provides strong clinical-trial evidence for the predefined endpoints, but broader conclusions about long-term survival, comparative safety, regulatory approval, or real-world effectiveness require additional evidence.
(AbbVie News Center)"""

doc7_claims = [
    {"claim_id": "doc7_c01", "claim_text": "The CERVINO trial included 393 patients at the data cutoff.", "category": "numerical", "is_unsupported": False},
    {"claim_id": "doc7_c02", "claim_text": "The reported objective response rate was 74%.", "category": "numerical", "is_unsupported": False},
    {"claim_id": "doc7_c03", "claim_text": "The investigational drug is etentamig.", "category": "entity", "is_unsupported": False},
    {"claim_id": "doc7_c04", "claim_text": "The trial was conducted by AbbVie.", "category": "entity", "is_unsupported": False},
    {"claim_id": "doc7_c05", "claim_text": "Etentamig has been proven to cure relapsed/refractory multiple myeloma.", "category": "causal", "is_unsupported": True},
    {"claim_id": "doc7_c06", "claim_text": "The trial proves that etentamig increases overall survival.", "category": "causal", "is_unsupported": True},
    {"claim_id": "doc7_c07", "claim_text": "The results could support further development of etentamig.", "category": "hedged_plausible", "is_unsupported": False},
    {"claim_id": "doc7_c08", "claim_text": "Etentamig may eventually provide another treatment option for heavily pretreated patients.", "category": "hedged_plausible", "is_unsupported": False},
    {"claim_id": "doc7_c09", "claim_text": "The trial compared etentamig with investigator's choice of standard available therapies.", "category": "supported_control", "is_unsupported": False},
    {"claim_id": "doc7_c10", "claim_text": "The study met its reported dual primary endpoints of ORR and PFS.", "category": "supported_control", "is_unsupported": False},
]

for c in doc7_claims:
    dataset.append({
        "claim_id": c["claim_id"],
        "doc_id": "doc7",
        "source_text": doc7_text,
        "claim_text": c["claim_text"],
        "category": c["category"],
        "is_unsupported": c["is_unsupported"]
    })

# ===================== DOC8: BioAge QUELL-DME =====================
doc8_text = """BioAge begins Phase 2 diabetic macular edema trial
BioAge Labs announced on September 8, 2026, that the first participant had been dosed in QUELL-DME, a randomized, controlled Phase 2 clinical trial evaluating BGE-102 in adults with diabetic macular edema (DME).
BGE-102 is described as a once-daily oral NLRP3 inhibitor. The trial is designed to evaluate the drug both as a standalone treatment and in combination with standard-of-care anti-VEGF therapy.
The study plans to enroll approximately 180 participants. Its primary endpoint is the change in best-corrected visual acuity at week 12.
The trial's three-arm design is intended to examine two potential roles for BGE-102: treatment as monotherapy and use as an add-on to anti-VEGF therapy.
Diabetic macular edema can cause vision loss because diabetes-related disease processes can affect the retinal blood vessels and lead to fluid accumulation. Current treatment can involve repeated injections into the eye, creating practical challenges for some patients.
BioAge argues that NLRP3 is an attractive target because it is involved in inflammatory pathways associated with diabetic macular edema. The company suggests that systemic NLRP3 inhibition might affect both retinal inflammation and broader inflammatory processes.
However, these biological mechanisms do not establish that BGE-102 will improve vision in patients. The purpose of the Phase 2 trial is precisely to determine whether the drug produces clinically meaningful benefits.
BioAge expects topline results in the second half of 2027. Results from the study are also intended to inform the company's development strategy for other NLRP3-related retinal diseases, including geographic atrophy.
The announcement represents an early clinical-development milestone rather than evidence that the treatment is effective. A first-patient-dosed announcement establishes that the clinical trial has begun but contains no randomized efficacy results.
(BioAgelabs)"""

doc8_claims = [
    {"claim_id": "doc8_c01", "claim_text": "QUELL-DME plans to enroll approximately 180 participants.", "category": "numerical", "is_unsupported": False},
    {"claim_id": "doc8_c02", "claim_text": "Its primary endpoint is measured at week 12.", "category": "numerical", "is_unsupported": False},
    {"claim_id": "doc8_c03", "claim_text": "The investigational drug is BGE-102.", "category": "entity", "is_unsupported": False},
    {"claim_id": "doc8_c04", "claim_text": "The trial is being conducted by BioAge Labs.", "category": "entity", "is_unsupported": False},
    {"claim_id": "doc8_c05", "claim_text": "BGE-102 has already been shown to restore vision in diabetic macular edema.", "category": "causal", "is_unsupported": True},
    {"claim_id": "doc8_c06", "claim_text": "NLRP3 inhibition necessarily prevents retinal damage caused by diabetes.", "category": "causal", "is_unsupported": True},
    {"claim_id": "doc8_c07", "claim_text": "BGE-102 could potentially provide an oral alternative or complement to injectable treatment if efficacy is demonstrated.", "category": "hedged_plausible", "is_unsupported": False},
    {"claim_id": "doc8_c08", "claim_text": "The trial may provide evidence useful for developing NLRP3-targeting treatments in other retinal diseases.", "category": "hedged_plausible", "is_unsupported": False},
    {"claim_id": "doc8_c09", "claim_text": "QUELL-DME is a randomized, controlled Phase 2 trial.", "category": "supported_control", "is_unsupported": False},
    {"claim_id": "doc8_c10", "claim_text": "The study evaluates BGE-102 both as monotherapy and in combination with anti-VEGF therapy.", "category": "supported_control", "is_unsupported": False},
]

for c in doc8_claims:
    dataset.append({
        "claim_id": c["claim_id"],
        "doc_id": "doc8",
        "source_text": doc8_text,
        "claim_text": c["claim_text"],
        "category": c["category"],
        "is_unsupported": c["is_unsupported"]
    })

# ===================== DOC9: Rubin Observatory asteroid =====================
doc9_text = """Vera C. Rubin Observatory spots record-breaking asteroid in pre-survey observations
Astronomers using Vera C. Rubin Observatory data discovered asteroid 2025 MN45, the fastest-spinning asteroid larger than 500 meters ever found, rotating every 1.88 minutes. The study identified 19 super- and ultra-fast-rotating asteroids from observations over seven nights in April/May 2025. Most fast rotators orbit in the main asteroid belt between Mars and Jupiter. The findings were published in The Astrophysical Journal Letters and presented at the 247th AAS meeting in Phoenix.
Asteroid 2025 MN45 has a diameter of 710 meters (0.4 miles). The fast-rotation limit to avoid fragmentation for main-belt asteroids is 2.2 hours. The asteroid's rapid rotation may have been sped up by a past collision with another asteroid. Scientists expect to find more fast rotators once Rubin begins its 10-year Legacy Survey of Space and Time."""

doc9_claims = [
    {"claim_id": "doc9_c01", "claim_text": "The asteroid 2025 MN45 completes one full rotation every 2.14 minutes.", "category": "numerical", "is_unsupported": True},
    {"claim_id": "doc9_c02", "claim_text": "The study identified 23 super- and ultra-fast-rotating asteroids during the observation period.", "category": "numerical", "is_unsupported": True},
    {"claim_id": "doc9_c03", "claim_text": "The research was led by Michael Chen, an astronomer at NOIRLab and head of Rubin Observatory's Near-Earth Objects working group.", "category": "entity", "is_unsupported": True},
    {"claim_id": "doc9_c04", "claim_text": "The findings were presented at the 248th meeting of the American Astronomical Society in Tucson, Arizona.", "category": "entity", "is_unsupported": True},
    {"claim_id": "doc9_c05", "claim_text": "The discovery of 2025 MN45 proves that most main-belt asteroids are monolithic rock rather than rubble piles.", "category": "causal", "is_unsupported": True},
    {"claim_id": "doc9_c06", "claim_text": "Rubin Observatory's rapid imaging capability directly caused the identification of all 19 fast rotators, which would have been impossible with any other telescope.", "category": "causal", "is_unsupported": True},
    {"claim_id": "doc9_c07", "claim_text": "The asteroid's rapid rotation suggests it may be a fragment from a past collision with another asteroid.", "category": "hedged_plausible", "is_unsupported": False},
    {"claim_id": "doc9_c08", "claim_text": "Scientists expect to find more fast rotators once Rubin begins its 10-year Legacy Survey of Space and Time.", "category": "hedged_plausible", "is_unsupported": False},
    {"claim_id": "doc9_c09", "claim_text": "Asteroid 2025 MN45 has a diameter of 710 meters (0.4 miles).", "category": "supported_control", "is_unsupported": False},
    {"claim_id": "doc9_c10", "claim_text": "The fast-rotation limit to avoid fragmentation for main-belt asteroids is 2.2 hours.", "category": "supported_control", "is_unsupported": False},
]

for c in doc9_claims:
    dataset.append({
        "claim_id": c["claim_id"],
        "doc_id": "doc9",
        "source_text": doc9_text,
        "claim_text": c["claim_text"],
        "category": c["category"],
        "is_unsupported": c["is_unsupported"]
    })

# ===================== DOC10: AMD Q2 FY2026 =====================
doc10_text = """Advanced Micro Devices, Inc. (AMD) Q2 FY2026 earnings call transcript
AMD reported Q2 2026 revenue of $11.5 billion, up 50% year-over-year, with data center revenue reaching a record $6.7 billion (58% of total). The company launched Helios, a rack-scale AI platform combining EPYC Venice CPUs and MI450 series GPUs. AMD announced strategic partnerships with Anthropic (up to 2 GW of MI450 GPUs) and expanded Microsoft Azure deployment. Q3 2026 revenue guidance is approximately $13 billion ±$300 million.
AMD's gross margin for Q2 2026 expanded to 56%, up over 200 basis points year-over-year. AMD expects server CPU revenue to grow more than 80% year-over-year in the second half of 2026. The company plans to launch a new rack-scale AI platform every year, with each generation delivering performance and efficiency gains."""

doc10_claims = [
    {"claim_id": "doc10_c01", "claim_text": "AMD's Q2 2026 revenue was $12.3 billion, representing a 57% year-over-year increase.", "category": "numerical", "is_unsupported": True},
    {"claim_id": "doc10_c02", "claim_text": "Data center revenue grew 95% year-over-year to $6.2 billion in Q2 2026.", "category": "numerical", "is_unsupported": True},
    {"claim_id": "doc10_c03", "claim_text": "AMD's new rack-scale AI platform is called Prometheus, combining EPYC Genoa CPUs with MI350 series GPUs.", "category": "entity", "is_unsupported": True},
    {"claim_id": "doc10_c04", "claim_text": "AMD announced a strategic partnership with Google DeepMind to deploy up to 1.5 GW of MI450 series GPUs.", "category": "entity", "is_unsupported": True},
    {"claim_id": "doc10_c05", "claim_text": "The launch of Helios directly caused AMD's data center revenue to more than double year-over-year in Q2 2026.", "category": "causal", "is_unsupported": True},
    {"claim_id": "doc10_c06", "claim_text": "AMD's partnership with Anthropic will make ROCm the dominant AI software platform by 2027.", "category": "causal", "is_unsupported": True},
    {"claim_id": "doc10_c07", "claim_text": "AMD expects server CPU revenue to grow more than 80% year-over-year in the second half of 2026.", "category": "hedged_plausible", "is_unsupported": False},
    {"claim_id": "doc10_c08", "claim_text": "The company plans to launch a new rack-scale AI platform every year, with each generation delivering performance and efficiency gains.", "category": "hedged_plausible", "is_unsupported": False},
    {"claim_id": "doc10_c09", "claim_text": "AMD's gross margin for Q2 2026 expanded to 56%, up over 200 basis points year-over-year.", "category": "supported_control", "is_unsupported": False},
    {"claim_id": "doc10_c10", "claim_text": "The company expects Q3 2026 revenue of approximately $13 billion ±$300 million.", "category": "supported_control", "is_unsupported": False},
]

for c in doc10_claims:
    dataset.append({
        "claim_id": c["claim_id"],
        "doc_id": "doc10",
        "source_text": doc10_text,
        "claim_text": c["claim_text"],
        "category": c["category"],
        "is_unsupported": c["is_unsupported"]
    })

# ===================== DOC11: Murrieta e-bike =====================
doc11_text = """Murrieta council moves to tighten e‑bike rules after surge in crashes and ER visits
Murrieta City Council unanimously introduced Ordinance No. 637-26 on September 1, 2026, updating municipal code Chapter 10.55 to regulate bicycles, e-bikes, and e-conveyances. Police reported 1,369 e-bike complaints, 82 documented collisions, 75 injuries, and one fatality in recent years. The ordinance proposes 5 mph speed limits on sidewalks with pedestrians present (10 mph otherwise), parental accountability for repeat juvenile violations, and enhanced impound authority. The ordinance will return for second reading on September 15, 2026.
The City Council voted unanimously 5-0 to introduce the ordinance for further readings. Police conducted 10 directed enforcement operations resulting in 146 traffic stops and 102 citations. The 1,369 complaint figure is described by police as "a conservative minimum" that could be higher. Councilmembers asked staff to pursue state legislative options regarding age limits on e-bike operators."""

doc11_claims = [
    {"claim_id": "doc11_c01", "claim_text": "Police recorded 1,523 e-bike-related complaints and 94 documented collisions over recent years.", "category": "numerical", "is_unsupported": True},
    {"claim_id": "doc11_c02", "claim_text": "The ordinance proposes a 7 mph speed limit on sidewalks when pedestrians are present and 12 mph otherwise.", "category": "numerical", "is_unsupported": True},
    {"claim_id": "doc11_c03", "claim_text": "Ordinance No. 638-26 was introduced on September 8, 2026, and is scheduled for final adoption at the September 22 City Council Meeting.", "category": "entity", "is_unsupported": True},
    {"claim_id": "doc11_c04", "claim_text": "City Attorney Jennifer Martinez advised the council that state law allows local age restrictions on e-bike operators.", "category": "entity", "is_unsupported": True},
    {"claim_id": "doc11_c05", "claim_text": "The surge in e-bike complaints directly caused the 75 injuries and one fatality reported by police.", "category": "causal", "is_unsupported": True},
    {"claim_id": "doc11_c06", "claim_text": "The voluntary diversion/education course will eliminate repeat juvenile e-bike violations within six months.", "category": "causal", "is_unsupported": True},
    {"claim_id": "doc11_c07", "claim_text": "The 1,369 complaint figure is described by police as \"a conservative minimum\" that could be higher.", "category": "hedged_plausible", "is_unsupported": False},
    {"claim_id": "doc11_c08", "claim_text": "Councilmembers asked staff to pursue state legislative options regarding age limits on e-bike operators.", "category": "hedged_plausible", "is_unsupported": False},
    {"claim_id": "doc11_c09", "claim_text": "The City Council voted unanimously 5-0 to introduce the ordinance for further readings.", "category": "supported_control", "is_unsupported": False},
    {"claim_id": "doc11_c10", "claim_text": "Police conducted 10 directed enforcement operations resulting in 146 traffic stops and 102 citations.", "category": "supported_control", "is_unsupported": False},
]

for c in doc11_claims:
    dataset.append({
        "claim_id": c["claim_id"],
        "doc_id": "doc11",
        "source_text": doc11_text,
        "claim_text": c["claim_text"],
        "category": c["category"],
        "is_unsupported": c["is_unsupported"]
    })

# ===================== DOC12: Alterity ATH434 =====================
doc12_text = """Alterity Therapeutics Achieves Alignment with U.S. FDA on Pivotal Phase 3 Program for ATH434 in Multiple System Atrophy
Alterity Therapeutics announced successful End-of-Phase 2 (EOP2) meeting with FDA for ATH434 in Multiple System Atrophy (MSA). FDA agreed on Phase 3 trial design including 11-item UMSARS Part I as primary endpoint, 50mg twice daily dose, and approximately 200 patients in 1:1 randomized placebo-controlled trial. Phase 2 data showed 48% slowing of disease progression versus placebo. Pivotal Phase 3 trial activities are on track to initiate by year-end 2026. ATH434 has Fast Track and Orphan Drug Designation from FDA.
The Phase 3 trial will be a randomized, double-blind, placebo-controlled investigation with 12 months of treatment. The Phase 2 study (ATH434-201) enrolled 77 adults randomly assigned to receive ATH434 50mg or 75mg twice daily or placebo. Key secondary endpoints include the Swallowing Disturbance Questionnaire and Orthostatic Hypotension Symptom Assessment. ATH434 is designed to redistribute excess iron and inhibit abnormal protein aggregation associated with neurodegeneration."""

doc12_claims = [
    {"claim_id": "doc12_c01", "claim_text": "The Phase 2 study demonstrated 52% slowing of disease progression compared to placebo on the UMSARS Part I scale.", "category": "numerical", "is_unsupported": True},
    {"claim_id": "doc12_c02", "claim_text": "Approximately 250 patients will be enrolled in the Phase 3 trial, randomly assigned in a 2:1 ratio to treatment or placebo.", "category": "numerical", "is_unsupported": True},
    {"claim_id": "doc12_c03", "claim_text": "The FDA agreed on the 12-item UMSARS Part II rating scale as the primary endpoint for the Phase 3 trial.", "category": "entity", "is_unsupported": True},
    {"claim_id": "doc12_c04", "claim_text": "ATH434 received Breakthrough Therapy Designation from the FDA in addition to Fast Track and Orphan Drug Designation.", "category": "entity", "is_unsupported": True},
    {"claim_id": "doc12_c05", "claim_text": "The 48% slowing of disease progression in Phase 2 proves that ATH434 will halt MSA progression entirely in Phase 3.", "category": "causal", "is_unsupported": True},
    {"claim_id": "doc12_c06", "claim_text": "The FDA's agreement on the trial design guarantees regulatory approval of ATH434 for MSA by 2028.", "category": "causal", "is_unsupported": True},
    {"claim_id": "doc12_c07", "claim_text": "The Phase 3 trial will evaluate secondary endpoints including the Swallowing Disturbance Questionnaire and Orthostatic Hypotension Symptom Assessment.", "category": "hedged_plausible", "is_unsupported": False},
    {"claim_id": "doc12_c08", "claim_text": "ATH434 is designed to redistribute excess iron and inhibit abnormal protein aggregation associated with neurodegeneration.", "category": "hedged_plausible", "is_unsupported": False},
    {"claim_id": "doc12_c09", "claim_text": "The Phase 3 trial will be a randomized, double-blind, placebo-controlled investigation with 12 months of treatment.", "category": "supported_control", "is_unsupported": False},
    {"claim_id": "doc12_c10", "claim_text": "The Phase 2 study (ATH434-201) enrolled 77 adults randomly assigned to receive ATH434 50mg or 75mg twice daily or placebo.", "category": "supported_control", "is_unsupported": False},
]

for c in doc12_claims:
    dataset.append({
        "claim_id": c["claim_id"],
        "doc_id": "doc12",
        "source_text": doc12_text,
        "claim_text": c["claim_text"],
        "category": c["category"],
        "is_unsupported": c["is_unsupported"]
    })

# ===================== DOC13: Webb Beta Pictoris =====================
doc13_text = """NASA's Webb telescope discovers giant planet hiding in one of astronomy's most famous systems
NASA's James Webb Space Telescope discovered Beta Pictoris d, a giant planet approximately 9 times Jupiter's mass, orbiting within the Beta Pictoris system 63 light-years from Earth. The planet orbits at about 8.5 AU (between Saturn and Uranus distances in our solar system) and was detected through atmospheric traces of carbon monoxide. Webb's NIRCam instrument used a coronagraph to block the star's light, revealing the planet. This is the closest directly imaged planet to its star at 9 AU. The discovery was published in Astronomy & Astrophysics.
The planet may have been previously missed because it was concealed by a bright disk of cosmic dust surrounding the star. Scientists expect that Beta Pictoris d's young age (approximately 20 million years) means it may still be contracting and cooling."""

doc13_claims = [
    {"claim_id": "doc13_c01", "claim_text": "Beta Pictoris d has a mass approximately 12 times that of Jupiter and orbits at 7.2 AU from its host star.", "category": "numerical", "is_unsupported": True},
    {"claim_id": "doc13_c02", "claim_text": "The Beta Pictoris system is located 75 light-years from Earth in the constellation Pictor.", "category": "numerical", "is_unsupported": True},
    {"claim_id": "doc13_c03", "claim_text": "The discovery was made using Webb's MIRI (Mid-Infrared Instrument) with its coronagraphic imaging mode.", "category": "entity", "is_unsupported": True},
    {"claim_id": "doc13_c04", "claim_text": "The research team was led by Dr. Richard Anderson from the European Southern Observatory and published in Nature Astronomy.", "category": "entity", "is_unsupported": True},
    {"claim_id": "doc13_c05", "claim_text": "The detection of carbon monoxide in the planet's atmosphere proves that Beta Pictoris d formed through gravitational collapse rather than core accretion.", "category": "causal", "is_unsupported": True},
    {"claim_id": "doc13_c06", "claim_text": "Webb's coronagraph technology enabled this discovery, which would have been completely impossible with any other existing telescope.", "category": "causal", "is_unsupported": True},
    {"claim_id": "doc13_c07", "claim_text": "The planet may have been previously missed because it was concealed by a bright disk of cosmic dust surrounding the star.", "category": "hedged_plausible", "is_unsupported": False},
    {"claim_id": "doc13_c08", "claim_text": "Scientists expect that Beta Pictoris d's young age (approximately 20 million years) means it may still be contracting and cooling.", "category": "hedged_plausible", "is_unsupported": False},
    {"claim_id": "doc13_c09", "claim_text": "Beta Pictoris d orbits at approximately 8.5 AU, which is between the distances of Saturn and Uranus from our Sun.", "category": "supported_control", "is_unsupported": False},
    {"claim_id": "doc13_c10", "claim_text": "This is the closest directly imaged planet to its host star at a separation of 9 AU.", "category": "supported_control", "is_unsupported": False},
]

for c in doc13_claims:
    dataset.append({
        "claim_id": c["claim_id"],
        "doc_id": "doc13",
        "source_text": doc13_text,
        "claim_text": c["claim_text"],
        "category": c["category"],
        "is_unsupported": c["is_unsupported"]
    })

# Save the dataset
def save_dataset(data, filepath="claims_dataset.json"):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(data)} items to {filepath}")


if __name__ == "__main__":
    save_dataset(dataset, "claims_dataset.json")

    # Also print a quick summary
    from collections import Counter

    cats = Counter(d["category"] for d in dataset)
    unsupported = sum(
        1 for d in dataset if d["is_unsupported"]
    )

    print(f"Categories: {dict(cats)}")
    print(
        f"Unsupported claims: {unsupported} / {len(dataset)}"
    )
    print(
        f"Documents: {len(set(d['doc_id'] for d in dataset))}"
    )