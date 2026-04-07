"""
ToS Corpus — Ground-Truth Dataset
====================================
A curated set of real-world style Terms of Service clauses with
human-verified labels.  Used by all three task graders.

Each entry contains:
  id          – unique clause identifier
  text        – the clause text the agent will see
  is_risky    – bool (ground truth for Task 1)
  category    – str (ground truth for Task 2)
  risk_score  – int 1-10 (ground truth component for Task 3)
  reasoning   – why it is/isn't risky (reference explanation)
  keywords    – key terms a good reasoning must mention (partial credit)
"""

from __future__ import annotations

from typing import Dict, List, Any

# ---------------------------------------------------------------------------
# 40 clauses — mix of risky and safe, covering all categories
# ---------------------------------------------------------------------------

CLAUSES: List[Dict[str, Any]] = [
    # ── PRIVACY ──────────────────────────────────────────────────────────
    {
        "id": "priv_001",
        "text": "We may share your personal information, including your name, email address, location, and browsing history, with third-party advertising partners without prior notice to you.",
        "is_risky": True,
        "category": "Privacy",
        "risk_score": 9,
        "reasoning": "Shares personal data with advertisers without user consent or notice.",
        "keywords": ["share", "third-party", "advertising", "personal", "data", "without notice"],
    },
    {
        "id": "priv_002",
        "text": "We collect usage data solely to improve our service and will never sell or share your personal information with any third party for marketing purposes.",
        "is_risky": False,
        "category": "Privacy",
        "risk_score": 2,
        "reasoning": "Explicitly prohibits selling or sharing data for marketing; low risk.",
        "keywords": ["never sell", "no third party", "improve service"],
    },
    {
        "id": "priv_003",
        "text": "By using this service, you consent to the collection and processing of your biometric data, including facial recognition data, for identity verification and fraud prevention.",
        "is_risky": True,
        "category": "Privacy",
        "risk_score": 9,
        "reasoning": "Biometric data collection is extremely sensitive and requires explicit opt-in.",
        "keywords": ["biometric", "facial recognition", "sensitive data"],
    },
    {
        "id": "priv_004",
        "text": "We retain your personal data for no longer than 90 days after account deletion, after which it is permanently destroyed.",
        "is_risky": False,
        "category": "Privacy",
        "risk_score": 2,
        "reasoning": "Clear data retention limit and deletion policy; user-friendly.",
        "keywords": ["retention", "deletion", "90 days", "destroyed"],
    },
    {
        "id": "priv_005",
        "text": "We may transfer your personal data to countries outside your jurisdiction, including countries that may not provide the same level of data protection as your home country.",
        "is_risky": True,
        "category": "Privacy",
        "risk_score": 7,
        "reasoning": "Cross-border transfer without guarantees of equivalent protection is a GDPR and privacy risk.",
        "keywords": ["transfer", "countries", "jurisdiction", "data protection"],
    },
    {
        "id": "priv_006",
        "text": "We use cookies and similar tracking technologies on our website to enhance your experience and remember your preferences.",
        "is_risky": False,
        "category": "Privacy",
        "risk_score": 3,
        "reasoning": "Standard cookie usage for UX; not unusually risky if consent is obtained.",
        "keywords": ["cookies", "tracking", "preferences"],
    },
    {
        "id": "priv_007",
        "text": "Your private messages and communications may be scanned and analysed by our automated systems and human reviewers for compliance, safety, and advertising personalisation.",
        "is_risky": True,
        "category": "Privacy",
        "risk_score": 9,
        "reasoning": "Reading private messages for advertising is a severe privacy violation.",
        "keywords": ["private messages", "scanned", "advertising", "human reviewers"],
    },

    # ── LIABILITY ─────────────────────────────────────────────────────────
    {
        "id": "liab_001",
        "text": "To the maximum extent permitted by law, we disclaim all warranties and shall not be liable for any direct, indirect, incidental, special, or consequential damages arising from your use of the service.",
        "is_risky": True,
        "category": "Liability",
        "risk_score": 8,
        "reasoning": "Blanket liability waiver removes all legal recourse for users harmed by the service.",
        "keywords": ["disclaim", "not liable", "damages", "waiver"],
    },
    {
        "id": "liab_002",
        "text": "You agree to indemnify and hold harmless the company, its officers, directors, and employees from any claims, damages, or expenses arising out of your use of the service or your violation of these terms.",
        "is_risky": True,
        "category": "Liability",
        "risk_score": 8,
        "reasoning": "Indemnification clause forces users to pay the company's legal costs.",
        "keywords": ["indemnify", "hold harmless", "claims", "expenses"],
    },
    {
        "id": "liab_003",
        "text": "Our total liability to you for any claim shall not exceed the amount you paid to us in the twelve months preceding the claim.",
        "is_risky": False,
        "category": "Liability",
        "risk_score": 4,
        "reasoning": "Limited liability cap tied to amount paid is common and somewhat reasonable.",
        "keywords": ["liability cap", "amount paid", "twelve months"],
    },
    {
        "id": "liab_004",
        "text": "You waive any right to bring a class action lawsuit against us. All disputes must be resolved individually through binding arbitration.",
        "is_risky": True,
        "category": "Liability",
        "risk_score": 9,
        "reasoning": "Waiving class action rights removes a major consumer protection mechanism.",
        "keywords": ["class action", "waive", "arbitration", "individual"],
    },
    {
        "id": "liab_005",
        "text": "We maintain comprehensive insurance and accept liability for direct damages caused by our negligence up to the full value of your subscription.",
        "is_risky": False,
        "category": "Liability",
        "risk_score": 2,
        "reasoning": "Accepts reasonable liability; user-friendly clause.",
        "keywords": ["accept liability", "negligence", "subscription value"],
    },

    # ── TERMINATION ───────────────────────────────────────────────────────
    {
        "id": "term_001",
        "text": "We reserve the right to suspend or terminate your account at any time, for any reason or no reason, without prior notice or explanation.",
        "is_risky": True,
        "category": "Termination",
        "risk_score": 9,
        "reasoning": "Arbitrary termination without any notice or reason is extremely unfair to users.",
        "keywords": ["terminate", "any reason", "no notice", "arbitrary"],
    },
    {
        "id": "term_002",
        "text": "If we terminate your account for violating our community guidelines, we will provide 30 days' notice and an opportunity to appeal the decision.",
        "is_risky": False,
        "category": "Termination",
        "risk_score": 2,
        "reasoning": "Fair termination policy with notice and appeals process.",
        "keywords": ["30 days notice", "appeal", "community guidelines"],
    },
    {
        "id": "term_003",
        "text": "Upon account termination for any reason, all data associated with your account, including your content, files, and purchase history, will be permanently deleted within 24 hours with no possibility of recovery.",
        "is_risky": True,
        "category": "Termination",
        "risk_score": 8,
        "reasoning": "Immediate, unrecoverable data deletion upon termination can cause significant harm.",
        "keywords": ["data deleted", "permanent", "24 hours", "no recovery"],
    },
    {
        "id": "term_004",
        "text": "You may cancel your account at any time. Upon cancellation, you will retain access until the end of your current billing period and can download your data for 30 days.",
        "is_risky": False,
        "category": "Termination",
        "risk_score": 1,
        "reasoning": "User-friendly cancellation with data portability window.",
        "keywords": ["cancel", "billing period", "download data", "30 days"],
    },
    {
        "id": "term_005",
        "text": "We may terminate your account immediately and without notice if we determine, in our sole discretion, that your continued use poses a risk to other users or to our business reputation.",
        "is_risky": True,
        "category": "Termination",
        "risk_score": 7,
        "reasoning": "Sole discretion termination for vague reasons provides no protection for users.",
        "keywords": ["sole discretion", "immediate", "without notice", "business reputation"],
    },

    # ── PAYMENTS ──────────────────────────────────────────────────────────
    {
        "id": "pay_001",
        "text": "All subscription fees are non-refundable under any circumstances, including if you cancel before the end of the billing period.",
        "is_risky": True,
        "category": "Payments",
        "risk_score": 7,
        "reasoning": "Blanket no-refund policy is unfair and may violate consumer protection laws.",
        "keywords": ["non-refundable", "no refund", "cancel", "billing period"],
    },
    {
        "id": "pay_002",
        "text": "We offer a 30-day money-back guarantee for all new subscriptions, no questions asked.",
        "is_risky": False,
        "category": "Payments",
        "risk_score": 1,
        "reasoning": "Generous refund policy; highly user-friendly.",
        "keywords": ["money-back", "guarantee", "30-day", "refund"],
    },
    {
        "id": "pay_003",
        "text": "We reserve the right to change our pricing at any time without notice. Continued use of the service after a price change constitutes your acceptance of the new pricing.",
        "is_risky": True,
        "category": "Payments",
        "risk_score": 8,
        "reasoning": "Unilateral price changes without notice and implied consent are highly unfair.",
        "keywords": ["price change", "without notice", "continued use", "acceptance"],
    },
    {
        "id": "pay_004",
        "text": "Your subscription will automatically renew at the end of each billing period. You will be notified 30 days before renewal and may cancel at any time.",
        "is_risky": False,
        "category": "Payments",
        "risk_score": 2,
        "reasoning": "Auto-renewal with advance notice and cancellation option is fair practice.",
        "keywords": ["auto-renew", "30 days", "notify", "cancel"],
    },
    {
        "id": "pay_005",
        "text": "We may charge additional fees for features, storage, or usage that exceeds limits described in your plan, without prior notification, charged immediately to your payment method on file.",
        "is_risky": True,
        "category": "Payments",
        "risk_score": 8,
        "reasoning": "Unexpected charges without notification are deceptive billing practices.",
        "keywords": ["additional fees", "without notification", "charged immediately"],
    },

    # ── CHANGES ───────────────────────────────────────────────────────────
    {
        "id": "chng_001",
        "text": "We may modify these Terms of Service at any time. Your continued use of the service after any modifications constitutes your acceptance of the updated terms.",
        "is_risky": True,
        "category": "Changes",
        "risk_score": 7,
        "reasoning": "Implicit acceptance of any changed terms with no opt-out violates informed consent.",
        "keywords": ["modify", "continued use", "acceptance", "any time"],
    },
    {
        "id": "chng_002",
        "text": "We will provide at least 30 days' notice before making any material changes to these Terms. You may cancel your account if you do not agree with the changes.",
        "is_risky": False,
        "category": "Changes",
        "risk_score": 2,
        "reasoning": "Advance notice with opt-out right is fair practice.",
        "keywords": ["30 days notice", "material changes", "cancel", "do not agree"],
    },
    {
        "id": "chng_003",
        "text": "We reserve the right to change, suspend, or discontinue any feature or the entire service at any time without liability or prior notice.",
        "is_risky": True,
        "category": "Changes",
        "risk_score": 8,
        "reasoning": "Service can be withdrawn at any time with no obligation to users.",
        "keywords": ["discontinue", "without liability", "without notice", "suspend"],
    },
    {
        "id": "chng_004",
        "text": "Updates to these Terms will be clearly communicated via email and in-app notifications. Changes will take effect 14 days after notification.",
        "is_risky": False,
        "category": "Changes",
        "risk_score": 1,
        "reasoning": "Proactive notification with adequate lead time is best practice.",
        "keywords": ["email notification", "14 days", "communicated"],
    },
    {
        "id": "chng_005",
        "text": "We may unilaterally alter the features, pricing, data policies, and terms of this agreement at any time. Changes become effective immediately upon posting.",
        "is_risky": True,
        "category": "Changes",
        "risk_score": 9,
        "reasoning": "Immediate unilateral changes across all aspects of the agreement is extremely unfair.",
        "keywords": ["unilaterally", "immediately", "posting", "alter"],
    },

    # ── MIXED / GENERAL ───────────────────────────────────────────────────
    {
        "id": "gen_001",
        "text": "By accepting these terms, you grant us an irrevocable, royalty-free, worldwide licence to use, reproduce, modify, adapt, publish, translate, and distribute any content you post on our platform.",
        "is_risky": True,
        "category": "Liability",
        "risk_score": 8,
        "reasoning": "Irrevocable IP licence transfer is a significant rights surrender.",
        "keywords": ["irrevocable", "royalty-free", "licence", "content", "worldwide"],
    },
    {
        "id": "gen_002",
        "text": "Users retain full ownership of all content they upload or create on our platform. We will only use your content to provide the service you requested.",
        "is_risky": False,
        "category": "Privacy",
        "risk_score": 1,
        "reasoning": "Full IP retention by user with narrow usage licence is very user-friendly.",
        "keywords": ["full ownership", "retain", "IP", "narrow licence"],
    },
    {
        "id": "gen_003",
        "text": "We may share your account data, including your full name, email, payment history, and usage patterns, with our parent company and all of its subsidiaries for marketing analytics.",
        "is_risky": True,
        "category": "Privacy",
        "risk_score": 8,
        "reasoning": "Wide sharing of sensitive data within corporate group for marketing is a privacy risk.",
        "keywords": ["share", "parent company", "subsidiaries", "marketing analytics", "payment"],
    },
    {
        "id": "gen_004",
        "text": "These Terms shall be governed by the laws of [Jurisdiction]. Any disputes must be brought exclusively in the courts of [Jurisdiction], subjecting international users to potentially costly out-of-country litigation.",
        "is_risky": True,
        "category": "Liability",
        "risk_score": 6,
        "reasoning": "Exclusive jurisdiction clause can make dispute resolution inaccessible for international users.",
        "keywords": ["jurisdiction", "exclusive courts", "international", "litigation"],
    },
    {
        "id": "gen_005",
        "text": "Our service is provided 'as-is' without any warranties, express or implied, including but not limited to merchantability, fitness for a particular purpose, or non-infringement.",
        "is_risky": True,
        "category": "Liability",
        "risk_score": 7,
        "reasoning": "As-is disclaimer removes all quality guarantees, shifting risk entirely to users.",
        "keywords": ["as-is", "no warranty", "merchantability", "fitness"],
    },
    {
        "id": "gen_006",
        "text": "You agree not to use our service for illegal activities. We reserve the right to cooperate fully with law enforcement agencies and to disclose your information when required by law.",
        "is_risky": False,
        "category": "Privacy",
        "risk_score": 3,
        "reasoning": "Standard law enforcement cooperation clause; required by law in many jurisdictions.",
        "keywords": ["law enforcement", "required by law", "cooperate"],
    },
    {
        "id": "gen_007",
        "text": "We reserve the right to access, monitor, and record all activity on your account at any time for purposes including but not limited to security, advertising optimisation, and product development.",
        "is_risky": True,
        "category": "Privacy",
        "risk_score": 8,
        "reasoning": "Broad surveillance of all account activity including for advertising is a serious privacy risk.",
        "keywords": ["monitor", "record", "all activity", "advertising", "surveillance"],
    },
    {
        "id": "gen_008",
        "text": "In the event of a data breach affecting your personal information, we will notify you within 72 hours and provide credit monitoring services at no cost to you.",
        "is_risky": False,
        "category": "Privacy",
        "risk_score": 1,
        "reasoning": "Fast breach notification with concrete remediation is a gold standard practice.",
        "keywords": ["breach notification", "72 hours", "credit monitoring"],
    },
    {
        "id": "gen_009",
        "text": "Children under 13 may use this service, and by allowing them to do so, parents/guardians agree to the collection of their child's personal data as described in our Privacy Policy.",
        "is_risky": True,
        "category": "Privacy",
        "risk_score": 9,
        "reasoning": "Collecting children's data requires explicit COPPA/GDPR-K compliance and highest level of protection.",
        "keywords": ["children", "under 13", "personal data", "COPPA"],
    },
    {
        "id": "gen_010",
        "text": "You acknowledge that you have read, understood, and agree to be bound by these Terms of Service.",
        "is_risky": False,
        "category": "Changes",
        "risk_score": 1,
        "reasoning": "Standard acceptance clause with no specific risk.",
        "keywords": ["acknowledge", "agree", "bound"],
    },
]

# ---------------------------------------------------------------------------
# Full document for Task 3 (Hard) - 3 pre-built documents
# ---------------------------------------------------------------------------

DOCUMENTS: List[Dict[str, Any]] = [
    {
        "id": "doc_social_media",
        "title": "SocialApp Terms of Service",
        "text": "\n\n".join([
            CLAUSES[0]["text"],   # priv_001 - risky
            CLAUSES[13]["text"],  # term_001 - risky
            CLAUSES[1]["text"],   # priv_002 - safe
            CLAUSES[7]["text"],   # liab_001 - risky
            CLAUSES[22]["text"],  # chng_001 - risky
            CLAUSES[20]["text"],  # pay_001  - risky
            CLAUSES[3]["text"],   # priv_004 - safe
            CLAUSES[28]["text"],  # gen_001  - risky
            CLAUSES[33]["text"],  # gen_006  - safe
            CLAUSES[36]["text"],  # gen_009  - risky
        ]),
        "ground_truth_risky_clauses": [
            {"clause_id": "priv_001", "category": "Privacy",     "risk_score": 9},
            {"clause_id": "term_001", "category": "Termination", "risk_score": 9},
            {"clause_id": "liab_001", "category": "Liability",   "risk_score": 8},
            {"clause_id": "chng_001", "category": "Changes",     "risk_score": 7},
            {"clause_id": "pay_001",  "category": "Payments",    "risk_score": 7},
            {"clause_id": "gen_001",  "category": "Liability",   "risk_score": 8},
            {"clause_id": "gen_009",  "category": "Privacy",     "risk_score": 9},
        ],
    },
    {
        "id": "doc_cloud_storage",
        "title": "CloudStore Terms of Service",
        "text": "\n\n".join([
            CLAUSES[4]["text"],   # priv_005 - risky
            CLAUSES[10]["text"],  # liab_004 - risky
            CLAUSES[11]["text"],  # liab_005 - safe
            CLAUSES[14]["text"],  # term_003 - risky
            CLAUSES[18]["text"],  # pay_003  - risky
            CLAUSES[21]["text"],  # pay_004  - safe
            CLAUSES[25]["text"],  # chng_003 - risky
            CLAUSES[30]["text"],  # gen_003  - risky
            CLAUSES[35]["text"],  # gen_008  - safe
        ]),
        "ground_truth_risky_clauses": [
            {"clause_id": "priv_005", "category": "Privacy",     "risk_score": 7},
            {"clause_id": "liab_004", "category": "Liability",   "risk_score": 9},
            {"clause_id": "term_003", "category": "Termination", "risk_score": 8},
            {"clause_id": "pay_003",  "category": "Payments",    "risk_score": 8},
            {"clause_id": "chng_003", "category": "Changes",     "risk_score": 8},
            {"clause_id": "gen_003",  "category": "Privacy",     "risk_score": 8},
        ],
    },
    {
        "id": "doc_subscription_app",
        "title": "SubApp Premium Terms of Service",
        "text": "\n\n".join([
            CLAUSES[6]["text"],   # priv_007 - risky
            CLAUSES[8]["text"],   # liab_002 - risky
            CLAUSES[15]["text"],  # term_004 - safe
            CLAUSES[19]["text"],  # pay_004  - safe
            CLAUSES[22]["text"],  # pay_005  - risky
            CLAUSES[26]["text"],  # chng_004 - safe
            CLAUSES[27]["text"],  # chng_005 - risky
            CLAUSES[31]["text"],  # gen_004  - risky
            CLAUSES[34]["text"],  # gen_007  - risky
        ]),
        "ground_truth_risky_clauses": [
            {"clause_id": "priv_007", "category": "Privacy",     "risk_score": 9},
            {"clause_id": "liab_002", "category": "Liability",   "risk_score": 8},
            {"clause_id": "pay_005",  "category": "Payments",    "risk_score": 8},
            {"clause_id": "chng_005", "category": "Changes",     "risk_score": 9},
            {"clause_id": "gen_004",  "category": "Liability",   "risk_score": 6},
            {"clause_id": "gen_007",  "category": "Privacy",     "risk_score": 8},
        ],
    },
]

# Convenience lookups
CLAUSE_BY_ID: Dict[str, Dict[str, Any]] = {c["id"]: c for c in CLAUSES}
DOCUMENT_BY_ID: Dict[str, Dict[str, Any]] = {d["id"]: d for d in DOCUMENTS}

# Only risky clauses (for Task 1 & 2 easy/medium sampling)
RISKY_CLAUSES = [c for c in CLAUSES if c["is_risky"]]
SAFE_CLAUSES  = [c for c in CLAUSES if not c["is_risky"]]

VALID_CATEGORIES = {"Privacy", "Liability", "Termination", "Payments", "Changes", "Other"}

# Category adjacency map — for partial credit in Task 2
ADJACENT_CATEGORIES = {
    "Privacy":     {"Privacy", "Liability", "Changes"},
    "Liability":   {"Liability", "Privacy", "Termination", "Changes"},
    "Termination": {"Termination", "Liability", "Payments"},
    "Payments":    {"Payments", "Termination", "Changes"},
    "Changes":     {"Changes", "Privacy", "Payments", "Liability"},
    "Other":       {"Other"},
}
