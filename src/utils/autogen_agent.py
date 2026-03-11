"""Custom AutoGen agent for gpt-oss compatibility.

``autogen`` is imported lazily so this module can be part of the package
without requiring the ``pyautogen`` dependency at import time.
"""

from .llm_wrapper import strip_after_message_marker, clean_messages_for_llm


def create_assistant_agent_gptoss(gpt_oss: bool = True):
    """
    Create AssistantAgent class for gpt-oss compatibility.
    
    Args:
        gpt_oss: If True, creates custom agent with message marker cleaning
        
    Returns:
        AssistantAgent class (custom or standard)
    
    Raises:
        ImportError: If ``autogen`` (pyautogen) is not installed.
    """
    try:
        import autogen
    except ImportError as exc:
        raise ImportError(
            "The 'pyautogen' package is required to use autogen agents. "
            "Install it with: pip install pyautogen"
        ) from exc

    if gpt_oss:
        class AssistantAgent_gptoss(autogen.AssistantAgent):
            """Custom AssistantAgent that cleans message markers for gpt-oss."""
            
            def generate_oai_reply(
                self,
                messages=None,
                sender=None,
                config=None,
                **kwargs,
            ):
                # 1) Clean incoming context BEFORE it hits the LLM
                if messages is not None:
                    messages = clean_messages_for_llm(messages)

                # Call the base implementation (this returns ok, reply)
                ok, reply = super(AssistantAgent_gptoss, self).generate_oai_reply(
                    messages=messages,
                    sender=sender,
                    config=config,
                    **kwargs,
                )

                # 2) Clean the reply we return
                if isinstance(reply, str):
                    reply = strip_after_message_marker(reply)
                elif isinstance(reply, dict):
                    content = reply.get("content", "")
                    reply["content"] = strip_after_message_marker(content)

                # 3) Clean history so future calls don't resend tags
                self._sanitize_history()

                return ok, reply

            def generate_reply(
                self,
                messages=None,
                sender=None,
                **kwargs,
            ):
                """This is what GroupChatManager calls: must return ONLY `reply`."""
                # If messages not provided (GroupChat often does this), pull from history
                if messages is None and sender is not None:
                    messages = self.chat_messages.get(sender, [])

                ok, reply = self.generate_oai_reply(
                    messages=messages,
                    sender=sender,
                    config=None,
                )

                # GroupChat only cares about the reply content
                return reply

            def _sanitize_history(self):
                """Clean chat_messages (visible groupchat history)."""
                # Clean chat_messages (visible groupchat history)
                for conv in getattr(self, "chat_messages", {}).values():
                    for m in conv:
                        c = m.get("content")
                        if isinstance(c, str):
                            m["content"] = strip_after_message_marker(c)

                # Clean internal _oai_messages (Autogen's per-sender cache)
                if hasattr(self, "_oai_messages"):
                    for sender, msgs in self._oai_messages.items():
                        for m in msgs:
                            c = m.get("content")
                            if isinstance(c, str):
                                m["content"] = strip_after_message_marker(c)
        
        return AssistantAgent_gptoss
    else:
        return autogen.AssistantAgent


# Lazy default export: only created when accessed, not at import time.
# Users should call create_assistant_agent_gptoss() directly, or access
# this attribute which will trigger the import on first use.
def __getattr__(name):
    if name == "AssistantAgent_gptoss":
        return create_assistant_agent_gptoss(gpt_oss=True)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

