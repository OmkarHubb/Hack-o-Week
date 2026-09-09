Step1: Install & Setup

!pip install -U "autogen-agentchat" "autogen-ext[openai]" python-dotenv

Step2: Define the model client (GROQ)

groq_client = OpenAIChatCompletionClient(

    model="llama-3.3-70b-versatile",

    base_url="https://api.groq.com/openai/v1",

    api_key=os.environ["GROQ_API_KEY"],

    model_info={

        "vision": False,

        "function_calling": True,

        "json_output": True,

        "family": "unknown",

    },

)

OR

Step2: Define the model client (OpenAI)

from autogen_ext.models.openai import OpenAIChatCompletionClient

openAI_client = OpenAIChatCompletionClient(model="gpt-4o")


Step3: Design the Agent Team

from autogen_agentchat.agents import AssistantAgent

 

market_researcher = AssistantAgent(

    "market_researcher",

    model_client=groq_client,

    system_message=(

        "You are a Market Research Analyst. Given a product/brand brief, identify: "

        "target audience segments, competitor positioning, and 2-3 relevant market trends. "

        "Be specific and concise — bullet points, no fluff."

    ),

)

 

strategist = AssistantAgent(

    "strategist",

    model_client=groq_client,

    system_message=(

        "You are a Marketing Strategist. Using the researcher's findings, propose a "

        "positioning statement, 3 core messaging pillars, and recommended channels "

        "(with rationale). Build directly on what the researcher provided."

    ),

)

 

copywriter = AssistantAgent(

    "copywriter",

    model_client=groq_client,

    system_message=(

        "You are a Copywriter. Turn the strategist's messaging pillars into: "

        "1 tagline, 1 short social post (under 280 chars), and 1 email subject line. "

        "Match tone to the brand described in the brief."

    ),

)

 

critic = AssistantAgent(

    "critic",

    model_client=groq_client,

    system_message=(

        "You are a Creative Director reviewing the full package (research, strategy, copy). "

        "Check for consistency, differentiation from competitors, and audience fit. "

        "If it's solid, respond with 'APPROVE'. Otherwise give one specific, actionable "

        "note for whoever should revise (name them)."

    ),

)

Note: - In step3, replace the groq_client with openAI_client if you want to work with OpenAI Model.

 

Step4: Write them into a team

from autogen_agentchat.conditions import TextMentionTermination

from autogen_agentchat.teams import RoundRobinGroupChat

from autogen_agentchat.ui import Console

 

termination = TextMentionTermination("APPROVE")

 

marketing_team = RoundRobinGroupChat(

    [market_researcher, strategist, copywriter, critic],

    termination_condition=termination,

    max_turns=12,

)

Step5: Run it

brief = """

Product: A subscription-based meal-prep kit for busy IT professionals in Pune, India.

Focus: high-protein, quick-cook meals under 20 minutes.

Goal: launch marketing strategy for the first 90 days.

"""

 

await Console(marketing_team.run_stream(task=brief))