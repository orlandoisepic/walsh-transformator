import os
import time

from typing import Generator

from prompt_toolkit.styles import Style
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit import PromptSession
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.completion import Completer, Completion, CompleteEvent, get_common_complete_suffix
from prompt_toolkit.enums import DEFAULT_BUFFER
from prompt_toolkit.filters import has_focus


def setup_cli() -> PromptSession:
    """
    Sets up the CLI by enabling a command history, command completion, and custom key bindings for "boop" sounds.
    :return: A prompt session.
    """
    # Path for persistent history
    HISTORY_FILE = os.path.expanduser("~/.cli_command_history")

    # File-based history
    history = FileHistory(HISTORY_FILE)

    # Commands. Should be
    # "cmd: None" for commands with no sub-commands;
    # "cmd: [args,...]" for commands with arguments;
    # "cmd: {subcmd: ...}" for nested commands.
    commands: dict[str, None | list | dict] = {
        "plot": {
            "coefficients": ["modified", "sorted", "first-n=", "original", "subtitle="],
            "transformation": ["modified", "subtitle="],
            "error": ["modified", "subtitle="],
            "base": ["subtitle="],
        },
        "discard": {
            "percentage": ["epsilon="],
            "absolute": ["epsilon="],
            "relative": ["epsilon="],
            "sparse": None,
        },
        "print": {
            "info": ["errors"]
        },
        "help": None,
        "exit": None,
    }

    completer = NestedCompleter(commands)

    # Key bindings for extra behavior
    kb = KeyBindings()

    suggestions = [""]  # object to be able to access it everywhere

    def _bottom_toolbar():
        # Returning "" hides the toolbar (no extra line).
        return suggestions[0]

    style = Style.from_dict({
        # Override the default bottom-toolbar style
        "bottom-toolbar": "noreverse",
    })

    @kb.add("up")
    def _(event):
        """
        "Boop" when the cursor is at the start of the buffer and "up-arrow" is pressed.
        """
        buf = event.current_buffer
        if buf.working_index is None or buf.working_index <= 0:
            event.app.output.bell()
        else:
            buf.history_backward()

    @kb.add("down")
    def _(event):
        """
        "Boop" when the cursor is at the end of the buffer and "down-arrow" is pressed.
        """
        buf = event.current_buffer
        if buf.working_index is None or buf.working_index >= len(buf._working_lines) - 1:
            event.app.output.bell()
        else:
            buf.history_forward()

    @kb.add("left")
    def _(event):
        """
        "Boop" when the cursor is at the left border and "left-arrow" is pressed.
        """
        buf = event.current_buffer
        if buf.cursor_position == 0:
            event.app.output.bell()
        else:
            buf.cursor_left(count=1)

    @kb.add("right")
    def _(event):
        """
        "Boop" when the cursor is at the right border and "right-arrow" is pressed.
        """
        buf = event.current_buffer
        if buf.cursor_position == len(buf.text):
            event.app.output.bell()
        else:
            buf.cursor_right(count=1)

    @kb.add("enter", filter=has_focus(DEFAULT_BUFFER))
    def _(event):
        """
        Hide the toolbar (i.e., the suggestions that were shown) when "enter" is pressed, i.e., a command is executed.
        """
        suggestions[0] = ""  # hide toolbar
        event.app.invalidate()
        # Let prompt_toolkit accept the input normally:
        event.app.current_buffer.validate_and_handle()

    DOUBLE_TAB_INTERVAL: float = 1  # seconds

    # State to detect consecutive Tabs and remember last completions
    double_tab_state: dict[str, float | tuple | bool] = {
        "time": 0.0,
        "candidates": None,
        "applied": False,
        "shown": False,
    }

    @kb.add("tab")
    def _(event):
        """
        Tab behavior should be as follows:

        First tab:  Complete the current input if it is unique; print suggestions if it is not unique.

        Second tab: If suggestions are shown, and it has been less time than DOUBLE_TAB_INTERVAL since the last tab,
        accept the first suggestion. If suggestions are shown, and it has been more time, do nothing.

        Next tabs:  If the suggestion was accepted, first show new suggestions, before accepting a new suggestion.

        Meanwhile, if a "faulty" tab-use is registered, e.g., there are no suggestions available for the current input;
        a "boop" sound is made.
        """
        now: float = time.time()

        buf = event.current_buffer
        completer = buf.completer

        # If no completer is attached, just bell
        if completer is None:
            event.app.output.bell()
            double_tab_state["time"] = 0.0
            double_tab_state["candidates"] = None
            suggestions[0] = ""
            return

        # Ask the completer for completions using CompleteEvent with explicit request
        complete_event = CompleteEvent(completion_requested=True)
        completions = list(completer.get_completions(buf.document, complete_event))
        names: list[str] = [c.text for c in completions]

        n: int = len(completions)

        if n == 0:
            double_tab_state["time"] = 0.0
            double_tab_state["candidates"] = None
            suggestions[0] = ""
            event.app.invalidate()
            event.app.output.bell()

        elif n == 1:
            # Single match -> accept it immediately.
            buf.apply_completion(completions[0])
            if not buf.document.text_after_cursor and not buf.text.endswith(" "):
                buf.insert_text(" ")
            double_tab_state["time"] = 0.0
            double_tab_state["candidates"] = None
            # Make sure completion-state is cleared (safe).
            buf.complete_state = None
            suggestions[0] = ""

        else:
            event.app.output.bell()
            same_candidates: bool = double_tab_state["candidates"] == tuple(names)
            if not same_candidates or double_tab_state["applied"]:
                # If either the candidates change or the previous completion was applied, we want to show the next suggestions
                double_tab_state["shown"] = False
            if now - double_tab_state["time"] <= DOUBLE_TAB_INTERVAL and same_candidates and not double_tab_state[
                "applied"]:
                buf.apply_completion(completions[0])
                buf.complete_state = None
                double_tab_state["applied"] = True
                # If the cursor is at end and there's no text after cursor, insert a space
                # so the completer will consider the next token (common shell behaviour).
                if not buf.document.text_after_cursor and not buf.text.endswith(" "):
                    buf.insert_text(" ")

                next_completions = list(
                    completer.get_completions(buf.document, CompleteEvent(completion_requested=True)))
                next_names: list[str] = [c.text for c in next_completions]
                if next_names:
                    common = get_common_complete_suffix(buf.document, next_completions)
                    if common:
                        buf.insert_text(common)
                    if not double_tab_state["applied"]:
                        # print_formatted_text("  ".join(next_names))
                        suggestions[0] = "  ".join(next_names)
                        double_tab_state["shown"] = True
                    event.app.invalidate()
                    double_tab_state["time"] = now
                    double_tab_state["candidates"] = tuple(next_names)
                else:
                    double_tab_state["time"] = 0.0
                    double_tab_state["candidates"] = None
                    suggestions[0] = ""

            else:
                # first tab (or new/different candidate set)
                # insert common suffix if any, print list
                common = get_common_complete_suffix(buf.document, completions)
                if common:
                    buf.insert_text(common)
                # with patch_stdout():
                if not double_tab_state["shown"]:
                    suggestions[0] = "  ".join(names)
                    double_tab_state["shown"] = True
                event.app.invalidate()
                double_tab_state["time"] = now
                double_tab_state["candidates"] = tuple(names)
                double_tab_state["applied"] = False

    session = PromptSession(
        history=history,
        enable_history_search=True,
        completer=completer,
        key_bindings=kb,
        complete_style=CompleteStyle.READLINE_LIKE,
        bottom_toolbar=_bottom_toolbar,
        style=style,
    )

    return session


class NestedCompleter(Completer):
    def __init__(self, commands: dict[str, list[str] | None | dict]):
        """
        Initialize the nested completer with a list of commands and their sub-commands.
        This means that the dictionary must list all commands and their sub-commands.
        If a command has value None, then it has no sub-commands.
        If it has a list as value, then the list must contain all possible sub-commands.
        If it has a dict as value, then each sub-command can have sub-commands as well.
        :param commands: A dictionary containing the commands to complete.
        """
        self.commands = commands

    def _traverse_to_node(self, path: list[str]) -> None | dict:
        """
        Follow the command-tree given by the path. If path is empty, then all top-level commands are returned.
        Otherwise, the path is followed until a node or a leaf is found.
        :param path: The (nested) command whose sub-command is to be completed (and thus the sub-command needs to be found).
        :return: The node found or None if the path doesn't exist or is a leaf.
        """
        node: dict = self.commands
        for token in path:
            if isinstance(node, dict) and token in node:
                node = node[token]
            else:
                # path cannot be followed
                return None
        return node

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Generator[Completion, None]:
        """
        Find completions for the given input document.
        :param document: The input text, along with cursor position.
        :param complete_event:
        :return: All possible completions.
        """
        text: str = document.text_before_cursor  # No lstrip() here as leading spaces matter for token positions
        # Split on whitespace
        words: list[str] = text.split()

        # Check if start is at new token
        at_new_token: bool = text.endswith(" ") or text == ""

        if at_new_token:
            # Path consists of all typed words (they are complete tokens).
            # We want to offer children of the last typed token (or top-level if none).
            path: list[str] = words  # may be [] if no words typed
            last_token: str = ""  # nothing to replace; insertion point is fresh token
        else:
            # Cursor is inside a token: last word is partial; path is everything before it
            path: list[str] = words[:-1]
            last_token: str = words[-1] if words else ""

        # Traverse commands according to path
        node = self._traverse_to_node(path)

        # If path traversal failed -> no completions
        if node is None:
            return

        # Determine candidate list based on node type
        if isinstance(node, dict):
            candidates = list(node.keys())
        elif isinstance(node, list):
            candidates = list(node)
        elif node is None:
            # Leaf command, no further completions
            return
        else:
            # Unexpected type
            return

        # Case-insensitive matching: compare lowercased
        last_lower = last_token.lower()
        start_pos = -len(last_token)  # -0 == 0 when last_token == ""

        for cand in candidates:
            if cand.lower().startswith(last_lower):
                # yield Completion with correct start position to replace the partial token
                yield Completion(cand, start_position=start_pos)
