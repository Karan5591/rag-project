EVALUATION_DATASET = [
    {
        "id": "T001",
        "category": "direct_relevant",
        "question": "How many unused PTO days can an employee roll over?",
        "expected": "answer",
        "expected_answer": "5",
    },
    {
        "id": "T002",
        "category": "direct_relevant",
        "question": "What does a flashing amber LED indicate?",
        "expected": "answer",
        "expected_answer": "fan failure or thermal threshold exceeded",
    },
    {
        "id": "T003",
        "category": "direct_relevant",
        "question": "What is the maximum throughput of the Edge Router?",
        "expected": "answer",
        "expected_answer": "18.4 Gbps",
        "expected_keywords": ["18.4"],
    },
    {
        "id": "T004",
        "category": "direct_relevant",
        "question": "What was the adjusted EBITDA for FY2025?",
        "expected": "answer",
        "expected_answer": "$98.4M",
    },
    {
        "id": "T005",
        "category": "direct_relevant",
        "question": "How long can EU customer PII remain in production databases?",
        "expected": "answer",
        "expected_answer": "180 days",
    },
    {
        "id": "T006",
        "category": "semantic_relevant",
        "question": "If an employee does not use all their leave, how many days can they carry forward?",
        "expected": "answer",
        "expected_answer": "5",
    },
    {
        "id": "T007",
        "category": "semantic_relevant",
        "question": "What condition causes the router's amber status light to flash?",
        "expected": "answer",
        "expected_answer": "fan failure or thermal threshold exceeded",
    },
    {
        "id": "T008",
        "category": "semantic_relevant",
        "question": "What is the Edge Router's maximum firewall processing capacity?",
        "expected": "answer",
        "expected_answer": "18.4 Gbps",
        "expected_keywords": ["18.4"],
    },
    {
        "id": "T009",
        "category": "irrelevant",
        "question": "Who is the Prime Minister of India?",
        "expected": "reject",
        "expected_answer": None,
    },
    {
        "id": "T010",
        "category": "irrelevant",
        "question": "What is the capital of France?",
        "expected": "reject",
        "expected_answer": None,
    },
    {
        "id": "T011",
        "category": "irrelevant",
        "question": "What is the company's maternity leave policy?",
        "expected": "reject",
        "expected_answer": None,
    },
    {
        "id": "T012",
        "category": "unsupported",
        "question": "What is the company's maternity leave policy for employees with more than 10 years of service?",
        "expected": "reject",
        "expected_answer": None,
    },

    # --------------------------------------------------------------
    # HARD CASES — from Initial_Testing_Report.docx.
    # These are the buried-detail / exception / table-value questions
    # that motivated building this RAG pipeline in the first place.
    # T001–T012 only test the easy, primary-fact version of each
    # topic; these test the specific detail that was actually failing.
    # `expected_keywords`: ALL must appear (case-insensitive) in the
    # retrieved chunk content for the case to pass.
    # --------------------------------------------------------------
    {
        "id": "T013",
        "category": "hard_detail",
        "question": "What is the deadline for obtaining a written exception to the PTO rollover forfeiture rule, and who must approve it?",
        "expected": "answer",
        "expected_answer": "December 15, VP of HR",
        "expected_keywords": ["December 15", "VP of HR"],
    },
    {
        "id": "T014",
        "category": "hard_detail",
        "question": "I'm a Tier 2 employee traveling to Europe for a meeting. What flight class and hotel budget am I allowed?",
        "expected": "answer",
        "expected_answer": "Business Class, $500/night",
        "expected_keywords": ["Business Class", "$500"],
    },
    {
        "id": "T015",
        "category": "hard_detail",
        "question": "Which region had the highest YoY growth rate in FY2025, and what was its total revenue?",
        "expected": "answer",
        "expected_answer": "APAC, 22.5%, $74.6M",
        "expected_keywords": ["APAC", "22.5%", "74.6"],
    },
    {
        "id": "T016",
        "category": "hard_detail",
        "question": "At what internal core temperature does the Edge Router trigger a fan failure warning?",
        "expected": "answer",
        "expected_answer": "75°C",
        "expected_keywords": ["75"],
    },
]