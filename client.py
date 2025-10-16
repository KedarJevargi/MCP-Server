import asyncio
import json
from typing import Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# For Ollama integration
import ollama

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.available_tools = []
        self.client_context = None
        self.session_context = None

    async def connect_to_server(self, server_script_path: str):
        """Connect to the MCP server"""
        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
            env=None
        )

        # Use context manager properly
        self.client_context = stdio_client(server_params)
        self.stdio, self.write = await self.client_context.__aenter__()

        self.session_context = ClientSession(self.stdio, self.write)
        self.session = await self.session_context.__aenter__()

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        self.available_tools = response.tools
        print(f"✅ Connected to MCP Server\n")

    async def process_tool_call(self, tool_name: str, tool_args: dict) -> str:
        """Execute a tool call on the MCP server"""
        if not self.session:
            raise RuntimeError("Not connected to server")

        result = await self.session.call_tool(tool_name, tool_args)
        return result.content[0].text

    def get_tools_for_llm(self) -> str:
        """Convert MCP tools to natural language description"""
        tools_desc = []
        for tool in self.available_tools:
            desc = f"- {tool.name}: {tool.description}"
            tools_desc.append(desc)
        return "\n".join(tools_desc)

    async def make_natural_response(self, user_query: str, raw_data: str) -> str:
        """Convert raw JSON data into natural, student-friendly response"""

        prompt = f"""You are a friendly and helpful AI assistant for BMS College of Engineering students. Your name is BMSCE Assistant and you're here to help students with information about college events, notifications, and academic content.

PERSONALITY:
- Be warm, friendly, and approachable like a helpful senior student
- Use casual but respectful language
- Show enthusiasm about college events and achievements
- Be encouraging and supportive
- Keep responses concise but informative
- Use emojis occasionally to be more engaging (but don't overdo it)

TASK:
A student asked: "{user_query}"

You retrieved this data:
{raw_data}

Now, present this information in a natural, conversational way. DO NOT mention that you got this from a database or API. Just present it as if you naturally know this information.

FORMATTING GUIDELINES:
- Use clear numbering (1., 2., 3.) for lists
- Include relevant dates naturally in the text
- Group related information together
- If there are many items, you can summarize or highlight the most important ones
- End with a friendly closing line if appropriate

Remember: Be natural, be friendly, be helpful! Act like a knowledgeable student helping another student.

Your response:"""

        response = ollama.generate(
            model='mistral:7b',
            prompt=prompt,
            options={
                "temperature": 0.8,
                "top_p": 0.9,
            }
        )

        return response['response'].strip()

    async def chat_with_mistral(self, user_message: str):
        """Chat with Mistral using MCP tools"""

        # Corrected prompt with explicit formats for each tool
        decision_prompt = f"""You are a precise tool selector. Your task is to analyze the user's question and choose the most appropriate tool from the list below.

# Available Tools & Formats:
1.  **get_latest_news**: Use for general inquiries about news, events, festivals, workshops, or what's happening.
    - Format: {{"tool": "get_latest_news", "arguments": {{}}}}

2.  **get_college_notifications**: Use for official notices, circulars, announcements, and deadlines.
    - Format: {{"tool": "get_college_notifications", "arguments": {{}}}}

3.  **query_knowledge_base**: Use to search for specific information like syllabus, student details, or specific people.
    - Format: {{"tool": "query_knowledge_base", "arguments": {{"query_text": "the user's search query"}}}}

4.  **none**: Use for greetings, thank yous, or conversational chat.
    - Format: {{"tool": "none"}}

# User Question:
"{user_message}"

# Instructions:
-   Analyze the user's question carefully.
-   Choose the single best tool that matches the user's intent.
-   **Respond with ONLY the JSON object in the exact format specified for the chosen tool.**

Your JSON response:"""

        # Get tool decision with lower temperature for consistency
        decision_response = ollama.generate(
            model='mistral:7b',
            prompt=decision_prompt,
            options={"temperature": 0.1, "top_p": 0.5}
        )

        tool_call = self._extract_tool_call(decision_response['response'])

        if tool_call and tool_call.get('tool') != 'none':
            tool_name = tool_call['tool']
            tool_args = tool_call.get('arguments', {})

            try:
                # Execute tool
                raw_data = await self.process_tool_call(tool_name, tool_args)

                # Convert to natural response
                natural_response = await self.make_natural_response(user_message, raw_data)
                print(f"{natural_response}\n")

            except Exception as e:
                # Improved debugging
                print(f"DEBUG: An error occurred while calling the tool: {e}")
                print(f"Oops! I had trouble getting that information. Could you try asking in a different way? 😊\n")
        else:
            # General conversation without tools
            chat_prompt = f"""You are a friendly AI assistant for BMS College of Engineering students.

User: {user_message}

Respond in a warm, helpful, and student-friendly way. Keep it concise and natural.

Your response:"""

            response = ollama.generate(
                model='mistral:7b',
                prompt=chat_prompt,
                options={"temperature": 0.8}
            )

            print(f"{response['response'].strip()}\n")

    def _extract_tool_call(self, text: str) -> Optional[dict]:
        """Extract tool call from LLM response"""
        try:
            # Remove markdown code blocks if present
            text = text.replace('```json', '').replace('```', '').strip()

            # Look for JSON object
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start:
                json_str = text[start:end]
                tool_call = json.loads(json_str)
                return tool_call
        except:
            pass
        return None

    async def close(self):
        """Close the connection"""
        if hasattr(self, 'session_context') and self.session_context:
            await self.session_context.__aexit__(None, None, None)
        if hasattr(self, 'client_context') and self.client_context:
            await self.client_context.__aexit__(None, None, None)


async def main():
    client = MCPClient()

    print("\n" + "🎓 " * 20)
    print("\n   Welcome to BMSCE Assistant! 🤖")
    print("   Your friendly AI helper for all things BMS College\n")
    print("🎓 " * 20 + "\n")

    # Connect to your MCP server
    await client.connect_to_server("main.py")

    print("💬 Hey there! Ask me about college events, notifications, or anything else!")
    print("   Type 'quit' or 'exit' when you're done.\n")
    print("─" * 60 + "\n")

    try:
        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                print("\n👋 See you later! Have an awesome day! 🌟\n")
                break

            if not user_input:
                continue

            print()
            await client.chat_with_mistral(user_input)
            print("─" * 60 + "\n")

    except KeyboardInterrupt:
        print("\n\n👋 Catch you later! Take care! 🌟\n")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())