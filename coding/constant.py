JOB_DEFINITION = {
    "DOCUMENT_TASK": f"Handles all actions related to individual documents or files.",
    "SELF_INTRODUCE": """Provides an introduction or detailed information about the agent when the user requests information about you. e.g. 'What can you do for me?', 'Introduce yourself', 'What can I do?' """,
    "LEARNING_TASK": [
        f"Handles actions where human want agent to learn from human inputs, including definition, feedback, clarification, corrections, or explanations from human inputs. The aim is to improve the agent's understanding, knowledge, or database over time through reinforcement learning, enabling it to adapt and enhance responses for future interactions.",
        "Human express opinions, ideas, thoughts."],
    "REPLY_TASK": "Processes messages that begin with '[REPLY_TASK]', indicating that the response should follow the user's prompt exactly as specified, without any additional instructions or context.",
    "USER_PERSONA": "Manage user's profile settings or questions about user identity, e.g. asking user's name, profile; setting or modifying the user's name, preferences, or traits, and adjusts responses accordingly.",
    "AGENT_PERSONA": """Manage agent's profile settings or questions about agent identity, e.g. asking agent's name, "Who are you?", "What's your name?", profile; setting or modifying the agent's name, personality traits, or behavior. """,
    "OPENING_MSG": "Processes messages that begin with '[OPENING_MSG]:', creates greeting messages when the considering context and user preferences.",
    "SOCIAL_NETWORK": ["Handles actions and questions related to activities involving friends, social networks, communities, or groups. This includes setting or managing group notifications, integrating tools to deliver responses about recent group activities, member lists, group information, documents in the groups, time inquiries, notifications, or messaging groups.","Handles actions and questions related to social network, relationship, messages, services."],
    "ASK_QUESTION": "Handles general Q&A from the user in any language, unrelated to operations to documents, introductions, social network, relationship, messages, or expert consultations.",
    "FALLBACK_TASK": "Politely refuses requests that cannot be fulfilled or are not defined in the agent's job definitions, ensuring respectful and appropriate responses.",
}

RESPONSE_FORMAT = {
    "JOB": "[Job classification for the prompt]",
    "PROMPT": "[Input from the user]"
}

EXPERTS_LIST = {
    "EXPERTS": [
      {
        "NAME": "Gild",
        "EMAIL": "gild@me",
        "URL": "www.nthu.edu.tw",
        "DISCIPLINE": "Digital Sociology",
        "CHARACTERISTIC": "Curious and analytical",
        "DESCRIPTION": "Gild examines how digital platforms influence social structures, identities, and collective behavior.",
        "INTEREST": "Online communities, data ethics, and social network analysis"
      },
      {
        "NAME": "Professor Furen",
        "EMAIL": "furen.lin@example.com",
        "URL": "www.furen.lin.com",
        "DISCIPLINE": "Information Systems Strategy",
        "CHARACTERISTIC": "Strategic and systems-oriented",
        "DESCRIPTION": "Furen researches how organizations leverage information systems to gain competitive advantage and enhance decision-making.",
        "INTEREST": "IT governance, digital transformation, and enterprise architecture"
      },
      {
        "NAME": "Brian Smith",
        "EMAIL": "brian.smith@example.org",
        "URL": "www.briansmith.dev",
        "DISCIPLINE": "Technology and Society",
        "CHARACTERISTIC": "Critical and forward-thinking",
        "DESCRIPTION": "Brian studies the societal impacts of emerging technologies, focusing on power, privacy, and inclusivity.",
        "INTEREST": "Tech policy, surveillance studies, and public interest technology"
      },
      {
        "NAME": "Carla Gomez",
        "EMAIL": "carla.gomez@domain.net",
        "URL": "www.carladesigns.net",
        "DISCIPLINE": "Human-Computer Interaction (HCI)",
        "CHARACTERISTIC": "Empathetic and research-driven",
        "DESCRIPTION": "Carla investigates how humans interact with digital systems, emphasizing usability and cultural context.",
        "INTEREST": "User research, interface design, and inclusive technology"
      },
      {
        "NAME": "Daniel Wu",
        "EMAIL": "daniel.wu@mail.com",
        "URL": "www.danielwu.tech",
        "DISCIPLINE": "Computational Social Science",
        "CHARACTERISTIC": "Innovative and data-savvy",
        "DESCRIPTION": "Daniel uses computational tools to analyze large-scale social phenomena, including opinion dynamics and misinformation.",
        "INTEREST": "Agent-based modeling, social simulations, and behavioral analytics"
      }
    ]
}

TEXTBOOK_LIST = {
    "TEXTBOOKS" : [
    {
      "TITLE": "The Digital Society",
      "AUTHOR": "Deborah Lupton",
      "DISCIPLINE": "Digital Sociology",
      "DESCRIPTION": "Explores how digital technologies shape contemporary social life, from identity to politics.",
      "RELATED_EXPERT": "Gild"
    },
    {
      "TITLE": "Strategic Information Systems Planning",
      "AUTHOR": "Robert D. Galliers",
      "DISCIPLINE": "Information Systems Strategy",
      "DESCRIPTION": "Covers frameworks and methodologies for aligning IT with organizational strategy.",
      "RELATED_EXPERT": "Professor Furen"
    },
    {
      "TITLE": "Technology and Society: Building Our Sociotechnical Future",
      "AUTHOR": "Deborah G. Johnson & Jameson M. Wetmore",
      "DISCIPLINE": "Technology and Society",
      "DESCRIPTION": "Analyzes how social, political, and cultural forces shape technology and vice versa.",
      "RELATED_EXPERT": "Brian Smith"
    },
    {
      "TITLE": "Human-Computer Interaction",
      "AUTHOR": "Alan Dix et al.",
      "DISCIPLINE": "Human-Computer Interaction (HCI)",
      "DESCRIPTION": "A comprehensive introduction to theories, methods, and design principles in HCI.",
      "RELATED_EXPERT": "Carla Gomez"
    },
    {
      "TITLE": "Computational Social Science",
      "AUTHOR": "Claudia Wagner et al.",
      "DISCIPLINE": "Computational Social Science",
      "DESCRIPTION": "Discusses how computational methods can model and analyze social phenomena.",
      "RELATED_EXPERT": "Daniel Wu"
    },
    {
      "TITLE": "Networks, Crowds, and Markets",
      "AUTHOR": "David Easley & Jon Kleinberg",
      "DISCIPLINE": "Social Network Analysis",
      "DESCRIPTION": "An interdisciplinary look at how networks shape economics and behavior.",
      "RELATED_EXPERT": "Gild"
    },
    {
      "TITLE": "IT Governance: How Top Performers Manage IT Decision Rights",
      "AUTHOR": "Jeanne W. Ross & Peter Weill",
      "DISCIPLINE": "Information Systems",
      "DESCRIPTION": "Practical guidance on structuring IT decision-making for organizational benefit.",
      "RELATED_EXPERT": "Professor Furen"
    },
    {
      "TITLE": "Surveillance Capitalism",
      "AUTHOR": "Shoshana Zuboff",
      "DISCIPLINE": "Tech Policy / Ethics",
      "DESCRIPTION": "Critically examines how companies exploit personal data under modern capitalism.",
      "RELATED_EXPERT": "Brian Smith"
    },
    {
      "TITLE": "Inclusive Design for a Digital World",
      "AUTHOR": "Regine Gilbert",
      "DISCIPLINE": "UX and Accessibility",
      "DESCRIPTION": "Guides designers on creating accessible and inclusive digital experiences.",
      "RELATED_EXPERT": "Carla Gomez"
    },
    {
      "TITLE": "Agent-Based Models of Geographical Systems",
      "AUTHOR": "A. Heppenstall et al.",
      "DISCIPLINE": "Agent-Based Modeling",
      "DESCRIPTION": "Covers the use of ABMs in understanding complex social and spatial processes.",
      "RELATED_EXPERT": "Daniel Wu"
    },
    {
      "TITLE": "Introduction to the Social Sciences",
      "AUTHOR": "Various",
      "DISCIPLINE": "General Knowledge",
      "DESCRIPTION": "Provides foundational understanding of sociology, economics, psychology, and anthropology.",
      "RELATED_EXPERT": "DEFAULT"
    }
  ]
}
