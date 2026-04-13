# After any changes to this file, re-run: python -m src.indexer
# This rebuilds the ChromaDB collection and BM25 index.

"""
ABC University — Study Programs & Admissions
Combined document set: 18 documents across 5 layers.

Layer 1  — General policies        (4 docs)  gen_001 – gen_004
Layer 2  — Program specific        (5 docs)  prog_cs_001, prog_ds_001,
                                              prog_ai_001, prog_mba_001, prog_ba_001
Layer 3  — Exceptions / overrides  (5 docs)  exc_001 – exc_005
Layer 4  — Guidance / scenarios    (2 docs)  ctx_001 – ctx_002
Layer 5  — FAQ / process           (2 docs)  faq_001 – faq_002

Priority system:
  3 = exception  (overrides everything — enforced in retrieval + prompt)
  2 = specific   (overrides general)
  1 = general    (default baseline)
  0 = faq        (supporting / procedural, lowest weight)

Metadata fields on every document:
  doc_id, doc_type, topics (list), program, audience, region, priority, valid_from
"""

DOCUMENTS = [

    # =========================================================================
    # LAYER 1 — GENERAL POLICIES
    # =========================================================================

    {
        "doc_id": "gen_001",
        "content": """
General Admission Requirements — All ABCStudy Programs

To be admitted to any undergraduate (Bachelor) program at ABC University,
applicants must hold a valid higher education entrance qualification. In Germany, this
is typically the Abitur or Fachhochschulreife. Equivalent qualifications from other
countries are accepted subject to recognition review.

For postgraduate (Master) programs, applicants must hold a completed Bachelor's degree
or an equivalent first academic degree from a recognised higher education institution.
The degree must be in a relevant subject area as defined per program. A minimum final
grade of 2.5 (German grading scale) or equivalent is required for most Master programs,
though some programs may apply stricter grade thresholds.

All applicants must demonstrate sufficient language proficiency:
- German-taught programs: minimum German language certificate at B2 level (e.g.
  TestDaF, DSH, Goethe-Zertifikat).
- English-taught programs: minimum IELTS 6.0 or TOEFL iBT 80, or equivalent.
  Native speakers of English or graduates of English-medium institutions may be exempt.

ABC operates a rolling admissions process with no fixed application deadlines for most
programs. Study can begin in any month of the year for online programs. Campus programs
typically start in March or October.

Applicants who do not meet standard requirements may be considered under the
professional experience pathway — see exception document exc_002 for details.
Applicants missing their final language certificate may in some cases receive
conditional admission — see exc_005.
        """.strip(),
        "metadata": {
            "doc_type": "general",
            "topics": ["admission", "eligibility", "language"],
            "program": "all",
            "audience": "all",
            "region": "all",
            "priority": 1,
            "valid_from": "2026-01-01"
        }
    },

    {
        "doc_id": "gen_002",
        "content": """
General Financing Options — All ABCStudy Programs

ABC University offers several ways to finance your studies:

1. Installment Payment Plans
   Tuition fees can be split into monthly installments at no additional cost.
   The standard plan divides the total program fee into equal monthly payments
   over the duration of the program. Early termination fees may apply if a
   student withdraws before completion.

2. BAföG (Federal Training Assistance)
   German residents and certain EU citizens may be eligible for BAföG state funding.
   ABC is a state-recognised university, meaning students enrolled in eligible programs
   can apply for BAföG through the relevant Studentenwerk. Online program students
   studying part-time are generally not eligible for BAföG. Full-time enrollments
   qualify. BAföG eligibility depends on personal income, parental income, and
   other factors assessed by the Studentenwerk.

3. ABCMerit Scholarship
   ABC offers merit-based scholarships covering partial tuition fees. Scholarships
   are competitive and awarded on a limited basis each intake cycle. Applicants
   must submit a separate scholarship application during the admissions process.
   Note: not all programs are eligible — see exc_004 for exclusions.

4. Need-Based Scholarship
   Available only to full-time students with demonstrated financial need. Part-time
   students are not eligible regardless of financial circumstances.

5. External Financing
   Students may apply for private education loans through partner banks. Employer
   sponsorship is also common, particularly for part-time and online students.
   ABC can provide official enrollment certificates to support employer reimbursement.

6. Early Enrollment Discount
   Applicants who complete enrollment more than 60 days before their intended start
   date may qualify for an early enrollment tuition discount. Percentage varies by
   program and intake cycle.
        """.strip(),
        "metadata": {
            "doc_type": "general",
            "topics": ["financing", "scholarship", "study_mode"],
            "program": "all",
            "audience": "all",
            "region": "all",
            "priority": 1,
            "valid_from": "2026-01-01"
        }
    },

    {
        "doc_id": "gen_003",
        "content": """
General Credit Transfer Policy — All ABCStudy Programs

ABC University recognises prior academic learning through its credit
transfer process. Students who have completed relevant coursework at another
recognised higher education institution may apply to have those credits counted
toward their ABC program.

General rules:
- Credit transfer requests must be submitted at the time of enrollment or within
  the first 30 days of study commencement.
- Transferred credits must correspond to modules offered within the student's
  enrolled program at ABC. Equivalency is assessed by the Academic Recognition Office.
- A maximum of 50% of a program's total credit volume may be transferred from
  external institutions. Students must complete at least 50% of their degree at ABC.
- Credits from non-accredited institutions are not eligible for transfer.
- Work experience alone does not qualify for academic credit transfer.

Grading conversion:
  Credits transferred from German institutions are accepted at face value.
  Credits from non-German institutions are converted using the Modified Bavarian
  Formula or an equivalent recognised conversion table.

Processing time:
  Credit transfer assessments typically take 4–6 weeks. Students may begin their
  studies while the assessment is pending.

Note: Some programs impose stricter transfer caps or exclude certain core modules.
See exc_003 for program-level overrides to this policy.
        """.strip(),
        "metadata": {
            "doc_type": "general",
            "topics": ["credits", "transfer"],
            "program": "all",
            "audience": "all",
            "region": "all",
            "priority": 1,
            "valid_from": "2026-01-01"
        }
    },

    {
        "doc_id": "gen_004",
        "content": """
Standard Application Process & Timeline — All ABCStudy Programs

Step 1 — Choose your program and study model
  Visit the ABC program catalog to select your program (Bachelor or Master) and
  study model (full-time online, part-time online, or campus-based).

Step 2 — Submit your online application
  Complete the online application at apply.iu.org. Required:
  - Personal details and contact information
  - Educational background and qualifications
  - Desired start date and study model
  No application fee is charged.

Step 3 — Upload supporting documents
  Required for all applicants:
  - Copy of highest educational qualification certificate
  - Official transcripts from all prior institutions
  - Proof of language proficiency (if applicable)
  - Valid government-issued ID or passport copy
  Additional documents may be required — see program-specific and exception docs.

Step 4 — Admissions review
  The Admissions Team reviews within 5–10 business days. You receive a conditional
  or unconditional offer by email. Conditional offers specify outstanding requirements.

Step 5 — Accept offer and enroll
  Accept via the admissions portal and complete the enrollment agreement including
  selecting your payment plan.

Step 6 — Begin studies
  Access to the ABC learning platform (MyABC) activates on your confirmed start date.

Timeline: typically 2–4 weeks for online programs, 4–8 weeks for campus programs.

Non-EU applicants should plan for additional lead time due to visa requirements.
See exc_001 and faq_002 for international applicant timelines.
        """.strip(),
        "metadata": {
            "doc_type": "general",
            "topics": ["application_process", "next_steps"],
            "program": "all",
            "audience": "all",
            "region": "all",
            "priority": 1,
            "valid_from": "2026-01-01"
        }
    },

    # =========================================================================
    # LAYER 2 — PROGRAM SPECIFIC
    # =========================================================================

    {
        "doc_id": "prog_cs_001",
        "content": """
BSc Computer Science — Program-Specific Admission Requirements

The Bachelor of Science in Computer Science at ABC is available as full-time or
part-time online, and at selected campuses.

Entry requirements:
  Standard ABC undergraduate admission requirements apply (see gen_001).
  No prior programming experience is formally required for admission, though
  students with no technical background are advised to complete ABC's free
  preparatory coding course before their start date.

Language of instruction:
  Available in German and English. Applicants must meet the language proficiency
  requirement corresponding to their chosen language track.

Credit transfer:
  Standard 50% cap applies. However, these core modules are excluded from
  external credit transfer regardless of prior study:
  - Algorithms and Data Structures (10 ECTS)
  - Software Engineering Fundamentals (5 ECTS)
  - ABC Capstone Project (10 ECTS)

Program duration:
  3 years full-time (180 ECTS). Part-time track: up to 6 years.

Scholarships:
  BSc CS students are eligible for the ABCSTEM Excellence Scholarship, covering
  up to 30% of tuition fees. Separate application required.
        """.strip(),
        "metadata": {
            "doc_type": "program",
            "topics": ["admission", "credits", "scholarship"],
            "program": "bsc_cs",
            "audience": "all",
            "region": "all",
            "priority": 2,
            "valid_from": "2026-01-01"
        }
    },

    {
        "doc_id": "prog_ds_001",
        "content": """
MSc Data Science — Program-Specific Admission Requirements

The Master of Science in Data Science at ABC is available fully online
(full-time and part-time) and at selected campuses.

Prior degree requirement:
  Applicants must hold a Bachelor's degree in: Computer Science, Mathematics,
  Statistics, Physics, Engineering, Information Systems, or a closely related
  technical discipline.
  Degrees in Business, Economics, Social Sciences, or Humanities may be considered
  if the applicant can demonstrate substantial quantitative coursework (minimum
  30 ECTS in mathematics, statistics, or programming) verified by official transcripts.
  Applicants from adjacent fields (e.g. economics, business analytics) may be
  accepted subject to bridging module requirements — see ctx_001 for guidance.
  Borderline cases — such as applicants from adjacent fields with limited
  quantitative coursework — may require a manual admissions review before
  a final eligibility determination is made.

Grade requirement:
  Minimum final grade of 2.3 (German scale). Stricter than the general ABCMaster
  threshold of 2.5 (gen_001). Applicants with grades between 2.4 and 2.5 may be
  considered with evidence of 2+ years relevant professional data experience.

Language of instruction:
  English only. Required: IELTS 6.5 or TOEFL iBT 90. Stricter than general
  ABC threshold of IELTS 6.0.

Credit transfer:
  Stricter cap of 30% of total program volume applies — overrides the general
  50% cap in gen_003. See exc_003 for full details. Core excluded modules:
  - Machine Learning in Practice (10 ECTS)
  - Data Engineering & Pipelines (5 ECTS)
  - Master's Thesis (30 ECTS)

Program duration:
  2 years full-time (120 ECTS). Part-time: up to 4 years.
        """.strip(),
        "metadata": {
            "doc_type": "program",
            "topics": ["admission", "eligibility", "credits"],
            "program": "msc_ds",
            "audience": "postgraduate",
            "region": "all",
            "priority": 2,
            "valid_from": "2026-01-01"
        }
    },

    {
        "doc_id": "prog_ai_001",
        "content": """
MSc Applied AI — Program-Specific Admission Requirements

The Master of Science in Applied AI at ABC is a technical postgraduate program
available fully online and at selected campuses.

Prior degree requirement:
  Applicants must hold a Bachelor's degree with a strong technical foundation.
  Accepted fields: Computer Science, Software Engineering, Mathematics, Electrical
  Engineering, or a closely related discipline with substantial programming and
  mathematics content.
  Unlike the MSc Data Science, the MSc Applied AI has limited flexibility for
  applicants from non-technical or adjacent fields (e.g. business, economics).
  Such applicants would typically not meet the technical prerequisites even with
  bridging modules, unless they have extensive documented programming experience.
  Applicants whose background does not clearly meet the technical prerequisites
  will typically be directed to admissions review rather than receiving an
  automatic rejection.

Grade requirement:
  Minimum final grade of 2.3 (German scale), consistent with MSc Data Science.

Language of instruction:
  English only. Required: IELTS 6.5 or TOEFL iBT 90.

Specific skill expectations:
  Prior exposure to Python or equivalent programming language is expected.
  Familiarity with linear algebra, probability, and calculus is assumed from
  day one of the program. Students without this background will struggle significantly.

Credit transfer:
  Standard 50% cap applies. Core modules excluded from transfer:
  - Deep Learning Fundamentals (10 ECTS)
  - AI Systems Design (10 ECTS)
  - MSc Thesis (30 ECTS)

Program duration:
  2 years full-time (120 ECTS). Part-time: up to 4 years.
        """.strip(),
        "metadata": {
            "doc_type": "program",
            "topics": ["admission", "eligibility"],
            "program": "msc_ai",
            "audience": "postgraduate",
            "region": "all",
            "priority": 2,
            "valid_from": "2026-01-01"
        }
    },

    {
        "doc_id": "prog_mba_001",
        "content": """
MBA — Program-Specific Guide

The MBA at ABC is a professionally-oriented postgraduate program available online
(full-time and part-time) and at selected campuses.

Entry requirements:
  - Completed Bachelor's degree in any field (business background preferred
    but not mandatory)
  - Minimum 1–2 years of relevant professional work experience expected.
    Applicants with strong academic backgrounds may be considered without
    work experience on a case-by-case basis.
  - English language: IELTS 6.0 or TOEFL iBT 80 (standard ABC threshold applies).

Scholarships and financing:
  The ABC merit scholarship does not apply to MBA students — see exc_004.
  Need-based scholarships also have restricted availability for MBA.
  Employer sponsorship and corporate tuition reimbursement are the most common
  financing routes for MBA students. ABC can issue invoices directly to employers.
  Installment payment plans are available as standard.

Credit transfer:
  Standard 50% cap applies. Business and management modules from prior study
  may be considered for transfer with subject equivalency review.

Program duration:
  1.5–2 years full-time (90–120 ECTS depending on track). Part-time: up to 4 years.

Note: Students expecting scholarship funding should carefully review exc_004
before applying, as the MBA has specific financing restrictions.
        """.strip(),
        "metadata": {
            "doc_type": "program",
            "topics": ["admission", "financing", "scholarship"],
            "program": "mba",
            "audience": "working_professional",
            "region": "all",
            "priority": 2,
            "valid_from": "2026-01-01"
        }
    },

    {
        "doc_id": "prog_ba_001",
        "content": """
BSc Business Administration — Program Guide

The Bachelor of Science in Business Administration is ABC's most enrolled program,
available fully online (full-time and part-time) and at selected campus locations
across Germany.

Admission requirements:
  Standard ABC undergraduate admission requirements apply (see gen_001).
  No specific prior subject background is required — the program is open to
  applicants from any secondary school qualification background.
  Language of instruction: available in both German and English. Applicants
  must meet the language proficiency requirement for their chosen track.

Study modes:
  Full-time online: standard 3-year duration (180 ECTS).
  Part-time online: up to 6 years. Most common choice for working students.
  Campus: available at selected locations, typically starting March or October.

Financing & Scholarship Options

In addition to general financing options (gen_002), the following
program-specific financing information applies to BSc BA students:

ABC Business Merit Scholarship:
  Available exclusively to BSc BA applicants.
  Covers up to 25% of total tuition fees.
  Eligibility:
  - Abitur or equivalent with final grade of 1.9 or better
  - Scholarship application submitted within 14 days of admissions offer
  - No outstanding conditions on admissions offer at time of application
  Cannot be combined with the general ABC need-based scholarship, but can be
  combined with employer sponsorship or BAföG.

Corporate Partner Tuition Reduction:
  Students employed by an ABC corporate partner company at enrollment may be
  eligible for a 10–20% tuition reduction. Students should check the corporate
  partner portal or contact their employer's HR department.

BAföG for BSc BA:
  Only full-time enrollment qualifies for BAföG. Part-time students are not
  eligible — consistent with general BAföG rules (gen_002).

Early enrollment discount:
  5% discount for enrollment completed 60+ days before start date.
        """.strip(),
        "metadata": {
            "doc_type": "program",
            "topics": ["financing", "scholarship", "admission"],
            "program": "bsc_ba",
            "audience": "all",
            "region": "all",
            "priority": 2,
            "valid_from": "2026-01-01"
        }
    },

    # =========================================================================
    # LAYER 3 — EXCEPTIONS & OVERRIDES
    # =========================================================================

    {
        "doc_id": "exc_001",
        "content": """
International Applicants — Non-EU Degree Recognition & Additional Requirements

Applicants holding qualifications obtained outside the European Union are subject
to additional requirements and a separate degree recognition process.

Degree recognition:
  Non-EU qualifications are assessed against the anabin database (maintained by
  the German Standing Conference of the Ministers of Education). Degrees from
  institutions with H+ status are generally accepted. H- institutions require
  additional documentation or may not be accepted.
  Applicants should check their institution's anabin status before applying. If
  not listed, ABC may request a Statement of Comparability from the German
  ENIC-NARIC centre (uni-assist assessment) — 6–10 weeks and a separate fee.
  In ambiguous cases where anabin status is unclear or documentation is
  incomplete, ABC's Academic Recognition Office will conduct a manual review
  before issuing a final admissions decision.

Additional documents required for non-EU applicants:
  - Certified German or English translation of all academic certificates
    (performed by a sworn/certified translator — uncertified translations not accepted)
  - Copy of valid passport (national ID alone is not sufficient)
  - APS certificate (mandatory for applicants from China, Vietnam, or Mongolia)
  - Proof of health insurance valid in Germany (for campus students)
  - Blocked account confirmation or proof of financial means (for visa applicants)

Credit transfer for non-EU applicants:
  Credits from non-EU institutions are subject to the standard transfer cap but
  require additional equivalency verification. Processing time: 6–8 weeks (vs.
  4–6 weeks for EU/German institutions). Plan accordingly.

Visa requirements:
  Non-EU students enrolling in campus programs must obtain a German student visa.
  ABC provides an official enrollment letter for visa applications upon completed
  enrollment. Online-only students studying outside Germany do not require a visa.
        """.strip(),
        "metadata": {
            "doc_type": "exception",
            "topics": ["admission", "visa", "credits"],
            "program": "all",
            "audience": "international",
            "region": "non_eu",
            "priority": 3,
            "valid_from": "2026-01-01"
        }
    },

    {
        "doc_id": "exc_002",
        "content": """
Professional Experience Pathway — Admission Without Standard Academic Qualifications

ABC offers an alternative admission pathway for applicants without standard academic
entry qualifications but with significant relevant professional experience.

IMPORTANT SCOPE LIMITATION:
  This pathway is available for UNDERGRADUATE (Bachelor) programs ONLY.
  Master programs require a completed Bachelor's degree. This exception does
  NOT apply to postgraduate admission under any circumstances.

Eligibility criteria:
  - Minimum 3 years of full-time professional work experience in a field
    relevant to the intended program
  - Completed vocational training (Berufsausbildung) or equivalent qualification,
    OR a portfolio demonstrating relevant competency
  - For technical programs (e.g. BSc CS): evidence of technical skills through
    portfolio, certifications, or a short assessment task

Application documents:
  - Detailed CV with employment history
  - Reference letter(s) from current or previous employer(s)
  - Vocational certificate or equivalent (if applicable)
  - Relevant professional certifications

Assessment timeline:
  Up to 15 business days (vs. 5–10 for standard applications). The Admissions
  Team may request a short interview or competency assessment.

Financing note:
  Students admitted via this pathway are eligible for all standard financing
  options (gen_002), including scholarships and installment plans. BAföG
  eligibility may be affected — consult the Studentenwerk directly.
        """.strip(),
        "metadata": {
            "doc_type": "exception",
            "topics": ["admission", "eligibility"],
            "program": "all",
            "audience": "working_professional",
            "region": "all",
            "priority": 3,
            "valid_from": "2026-01-01"
        }
    },

    {
        "doc_id": "exc_003",
        "content": """
Credit Transfer Exceptions & Program-Level Overrides

The general credit transfer policy (gen_003) establishes a default cap of 50%.
The following program-level overrides and exceptions apply:

Program-level cap overrides:
  - MSc Data Science: maximum transferable credits reduced to 30% of total
    program volume. This OVERRIDES the general 50% cap. Reason: the program
    has a high proportion of integrated project and lab modules that cannot
    be substituted externally.
  - BSc Computer Science: 50% cap applies as per general policy, but three
    core modules are excluded from transfer regardless of prior study (see
    prog_cs_001 for the excluded module list).
  - MSc Applied AI: 50% cap applies but core AI modules (30 ECTS in Deep
    Learning and AI Systems Design) are always excluded.
  - All other programs: standard 50% cap applies.

Non-accredited institutions:
  Credits from institutions not recognised by the German Accreditation Council
  or equivalent national body cannot be transferred under any circumstances,
  even if the student completed substantial coursework there.

MOOCs and micro-credentials:
  Credits from MOOCs (Coursera, edX, etc.) or micro-credential programs are
  NOT eligible for academic credit transfer regardless of platform reputation.
  They may be referenced in a portfolio for the professional pathway (exc_002)
  but carry no ECTS value at ABC.

Professional certifications:
  Industry certifications (AWS, Google, Microsoft etc.) do not qualify for
  credit transfer. They may strengthen an application but carry no ECTS value.

Retroactive requests:
  Credit transfer requests submitted after the 30-day enrollment window are
  not accepted. No exceptions to this deadline.
        """.strip(),
        "metadata": {
            "doc_type": "exception",
            "topics": ["credits", "transfer"],
            "program": "all",
            "audience": "all",
            "region": "all",
            "priority": 3,
            "valid_from": "2026-01-01"
        }
    },

    {
        "doc_id": "exc_004",
        "content": """
MBA Scholarship Exclusion Policy

This document overrides the general scholarship statements in gen_002 and
prog_mba_001 for MBA applicants specifically.

ABC Merit Scholarship — MBA exclusion:
  MBA students are EXCLUDED from the ABC merit scholarship program. This applies
  to all MBA tracks (full-time, part-time, online, campus). The exclusion is
  not based on merit or academic performance — it is a structural program policy.

Need-Based Scholarship — MBA restriction:
  Need-based scholarship seats for MBA students are extremely limited and awarded
  only in exceptional circumstances. MBA applicants should not plan their financing
  around need-based scholarship availability.

Why this distinction exists:
  The MBA is a professionally-oriented program with a different financing model.
  Employer sponsorship, corporate partner tuition reductions, and professional
  development budgets are the primary intended financing routes for MBA students.
  ABC actively supports employer billing arrangements for MBA students.

What MBA students CAN access:
  - Monthly installment payment plans (standard, no additional cost)
  - Corporate partner tuition reductions (10–20% if employer is a partner)
  - Early enrollment discount (5% if enrolled 60+ days before start)
  - External education loans via ABC partner banks
  - Employer sponsorship with official ABC invoicing

Applicants expecting scholarship funding should consider alternative programs
where merit scholarships are available (e.g. BSc BA, BSc CS, MSc DS, MSc AI).
        """.strip(),
        "metadata": {
            "doc_type": "exception",
            "topics": ["scholarship", "financing"],
            "program": "mba",
            "audience": "all",
            "region": "all",
            "priority": 3,
            "valid_from": "2026-01-01"
        }
    },

    {
        "doc_id": "exc_005",
        "content": """
Conditional Admission Policy — Missing or Pending Documents

ABC may issue conditional admission offers to applicants who have not yet
completed all required documentation at the time of application.

Cases where conditional admission may apply:
  - Applicant is finishing their current degree and does not yet have a
    final certificate (provisional transcripts accepted for initial review)
  - Applicant has not yet received their final language test score
    (IELTS/TOEFL result pending at time of application)
  - Applicant is awaiting certified translations of foreign documents
  - Applicant is awaiting uni-assist assessment result (non-EU cases)

Conditions of a conditional offer:
  - Outstanding documents must be submitted by a specified deadline, typically
    4–8 weeks after the conditional offer is issued
  - Enrollment is not confirmed until all conditions are met
  - ABC reserves the right to withdraw the offer if conditions are not fulfilled
    within the stated deadline

Important limitations:
  - Conditional admission does not guarantee final admission
  - Missing CORE eligibility requirements (e.g. no relevant degree at all
    for a Master's program) CANNOT be bypassed via conditional admission
  - Conditional admission is not the same as unconditional enrollment

Process:
  Applicants should clearly indicate in their application which documents are
  pending and provide an expected submission date. The Admissions Team will
  assess whether conditional admission is appropriate on a case-by-case basis.
        """.strip(),
        "metadata": {
            "doc_type": "exception",
            "topics": ["admission", "language"],
            "program": "all",
            "audience": "all",
            "region": "all",
            "priority": 3,
            "valid_from": "2026-01-01"
        }
    },

    # =========================================================================
    # LAYER 4 — USER CONTEXT
    # =========================================================================

    {
        "doc_id": "ctx_001",
        "content": """
Already Hold a Degree — Career Changers & Second-Degree Applicants

If you already hold a Bachelor's or higher degree and are applying to a new program
(either as a Master's applicant or as a career-changer into a second Bachelor's):

Applying to a Master's program with an existing Bachelor's:
  Your Bachelor's degree is your primary admission credential. Whether it is
  in a relevant field depends on the specific program:
  - MSc Data Science: adjacent degrees (economics, business analytics) may
    qualify with sufficient quantitative coursework. See prog_ds_001.
  - MSc Applied AI: limited flexibility — needs strong technical background.
    A business or social science degree is unlikely to qualify. See prog_ai_001.
  - MBA: any Bachelor's field is accepted; work experience matters more.

Applying to a second Bachelor's as a career changer:
  Your prior degree qualifies as equivalent to Abitur — you do not need to
  re-demonstrate secondary school qualifications.
  You may be eligible to transfer credits from your prior degree, but subject
  equivalency is the key constraint. A prior Business degree into BSc CS will
  have limited overlap — most technical core modules will not transfer.

Practical advice on credit transfer:
  Submit transcripts from your prior degree early. The Academic Recognition Office
  can provide a preliminary assessment before you formally enroll. This helps you
  plan your expected study duration accurately before committing.

  Career changers often overestimate transferability across very different fields.
  A realistic expectation is 10–25% credit transfer for a completely adjacent
  degree, not the theoretical 50% maximum.
        """.strip(),
        "metadata": {
            "doc_type": "guidance",
            "topics": ["credits", "admission", "eligibility"],
            "program": "all",
            "audience": "career_changer",
            "region": "all",
            "priority": 2,
            "valid_from": "2026-01-01"
        }
    },

    {
        "doc_id": "ctx_002",
        "content": """
Studying While Working — Part-Time Options & Practical Considerations

ABC is designed to accommodate students in full-time or part-time employment.

Study models for working students:
  - Part-time online: 15–20 hours of study per week. Program duration doubles
    (e.g. 6 years for a standard 3-year Bachelor's). Most common choice for
    working students.
  - Full-time online: ~30–40 hours per week. Feasible alongside part-time
    employment but challenging with full-time work.
  - Campus part-time: 1–2 fixed attendance days per week. Less flexible than
    online options.

Scheduling flexibility:
  ABC's online programs are fully asynchronous — no mandatory live sessions.
  Students access lectures and submit assessments on their own schedule within
  semester deadlines. Ideal for irregular hours, shift work, or different time zones.

BAföG important note:
  Part-time students are NOT eligible for BAföG regardless of financial
  circumstances. Full-time enrollment is required for BAföG eligibility.
  If BAföG is part of your financing plan, enroll full-time. See gen_002.

Employer sponsorship:
  Many working students have tuition covered by their employer. ABC issues
  official enrollment certificates and invoices addressed to employers to
  facilitate reimbursement. Some corporate partners have direct billing.

Professional experience pathway:
  If you have 3+ years of relevant work experience but lack the standard
  academic entry qualification, you may qualify for undergraduate admission
  through the professional experience pathway. See exc_002.
        """.strip(),
        "metadata": {
            "doc_type": "guidance",
            "topics": ["study_mode", "admission", "financing"],
            "program": "all",
            "audience": "working_professional",
            "region": "all",
            "priority": 2,
            "valid_from": "2026-01-01"
        }
    },

    # =========================================================================
    # LAYER 5 — FAQ / PROCESS (NEW vs original plan)
    # =========================================================================

    {
        "doc_id": "faq_001",
        "content": """
Start Dates, Deferrals & Next Steps — FAQ

When can I start?
  Online programs: rolling start — you can begin in any month of the year.
  Campus programs: standard intake in March and October.
  Most students find October and April the most common start months for
  planning purposes (academic year alignment), though online programs are
  flexible year-round.

Can I defer my start date?
  Deferral to the next intake is possible in most cases under the following
  conditions:
  - Request must be submitted before the original confirmed start date
  - Enrollment agreement must already be signed
  - One deferral per application is typically permitted
  - Tuition rates at the time of deferral apply, not the original enrollment rate

What are the immediate next steps after receiving an offer?
  1. Review your offer letter for any conditions
  2. Accept your offer via the admissions portal within the stated deadline
  3. Upload any outstanding documents
  4. Select and confirm your payment plan
  5. Sign the enrollment agreement
  6. Receive your MyABC learning platform access on your start date

When should I apply if I want a specific start month?
  Online programs: apply at least 3–4 weeks before your desired start to allow
  for admissions review and enrollment completion.
  Non-EU / international applicants: apply at least 3–4 months before your
  desired start to allow for document verification, uni-assist processing,
  and visa application. See faq_002 for international timeline details.
        """.strip(),
        "metadata": {
            "doc_type": "faq",
            "topics": ["application_process", "next_steps"],
            "program": "all",
            "audience": "all",
            "region": "all",
            "priority": 0,
            "valid_from": "2026-01-01"
        }
    },

    {
        "doc_id": "faq_002",
        "content": """
Visa & Enrollment Timeline — International Applicants FAQ

This FAQ is specifically for non-EU applicants planning to study on-campus
or who need a German student visa.

How long does the visa process take?
  German student visa processing typically takes 6–12 weeks after submission,
  depending on the German embassy in your country. Some countries (e.g. India,
  Nigeria) may have longer wait times due to appointment availability.

What do I need before applying for a visa?
  - Unconditional or fulfilled-condition admission letter from ABC
  - Completed enrollment agreement
  - Proof of financial means (blocked account: typically €11,208/year required)
  - Health insurance valid in Germany
  - All academic documents with certified translations

When should I apply for admission if I want to start in October?
  Recommended timeline working backwards from October start:
  - August: latest realistic date for visa approval
  - June–July: submit visa application to German embassy
  - May–June: complete ABC enrollment (fulfill all conditions)
  - April–May: receive unconditional admission offer from ABC
  - February–March: SUBMIT your ABC application

  This means non-EU campus applicants targeting October should apply to ABC
  no later than March of that year — ideally earlier.

Do online students need a visa?
  No. If you study fully online from outside Germany, you do not need a
  German student visa. You only need a visa if you physically study in Germany
  (campus program or extended on-campus residency).

What if my visa is delayed?
  Contact ABC Admissions. In some cases a deferral to the next intake can be
  arranged if the delay is due to embassy processing times outside your control.
  See faq_001 for deferral conditions.
        """.strip(),
        "metadata": {
            "doc_type": "faq",
            "topics": ["visa", "application_process"],
            "program": "all",
            "audience": "international",
            "region": "non_eu",
            "priority": 0,
            "valid_from": "2026-01-01"
        }
    },

]


# =============================================================================
# Validation — run: python data/documents.py
# =============================================================================
if __name__ == "__main__":
    expected_ids = [
        "gen_001", "gen_002", "gen_003", "gen_004",
        "prog_cs_001", "prog_ds_001", "prog_ai_001", "prog_mba_001", "prog_ba_001",
        "exc_001", "exc_002", "exc_003", "exc_004", "exc_005",
        "ctx_001", "ctx_002",
        "faq_001", "faq_002",
    ]
    actual_ids = [d["doc_id"] for d in DOCUMENTS]
    missing = [i for i in expected_ids if i not in actual_ids]
    extra   = [i for i in actual_ids if i not in expected_ids]

    assert not missing, f"Missing docs: {missing}"
    assert not extra,   f"Unexpected docs: {extra}"

    print(f"OK: All {len(DOCUMENTS)} documents present.\n")
    print(f"{'doc_id':<16} {'doc_type':<12} {'topics':<34} {'program':<12} {'priority'}")
    print("-" * 85)
    for doc in DOCUMENTS:
        m = doc["metadata"]
        topics_str = ", ".join(m["topics"])
        print(f"{doc['doc_id']:<16} {m['doc_type']:<12} {topics_str:<34} {m['program']:<12} {m['priority']}")
