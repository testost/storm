#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="dspy.*")
warnings.filterwarnings("ignore", message="Valid config keys have changed in V2")
warnings.filterwarnings("ignore", message=".*is deprecated.*")

import os
import sys
import cmd
from typing import Optional
import argparse
from colorama import Fore, Style, init
from datetime import datetime
import webbrowser
import re

from knowledge_storm import (
    STORMWikiRunnerArguments,
    STORMWikiRunner,
    STORMWikiLMConfigs,
)
from knowledge_storm.lm import OpenAIModel
from knowledge_storm.rm import YouRM

class StormCLI(cmd.Cmd):
    intro = f'''
{Fore.RED}   _____ _______ ____  _____  __  __ 
  / ____|__   __/ __ \|  __ \|  \/  |
 | (___    | | | |  | | |__) | \  / |
  \___ \   | | | |  | |  _  /| |\/| |
  ____) |  | | | |__| | | \ \| |  | |
 |_____/   |_|  \____/|_|  \_\_|  |_|{Style.RESET_ALL}
                                     
Welcome to STORM CLI! Type /help for commands or just enter your query.
'''
    prompt = f'{Fore.RED}S>{Style.RESET_ALL} '

    def __init__(self):
        super().__init__()
        init()  # Initialize colorama
        self.runner = None
        self.setup_storm()

    def setup_storm(self):
        """Initialize STORM with default configuration"""
        lm_configs = STORMWikiLMConfigs()
        openai_kwargs = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": 1.0,
            "top_p": 0.9,
        }

        ModelClass = OpenAIModel
        gpt_35_model_name = "gpt-3.5-turbo"
        gpt_4_model_name = "gpt-4"

        conv_simulator_lm = ModelClass(model=gpt_35_model_name, max_tokens=500, **openai_kwargs)
        question_asker_lm = ModelClass(model=gpt_35_model_name, max_tokens=500, **openai_kwargs)
        outline_gen_lm = ModelClass(model=gpt_4_model_name, max_tokens=400, **openai_kwargs)
        article_gen_lm = ModelClass(model=gpt_4_model_name, max_tokens=700, **openai_kwargs)
        article_polish_lm = ModelClass(model=gpt_4_model_name, max_tokens=4000, **openai_kwargs)

        lm_configs.set_conv_simulator_lm(conv_simulator_lm)
        lm_configs.set_question_asker_lm(question_asker_lm)
        lm_configs.set_outline_gen_lm(outline_gen_lm)
        lm_configs.set_article_gen_lm(article_gen_lm)
        lm_configs.set_article_polish_lm(article_polish_lm)

        engine_args = STORMWikiRunnerArguments(
            output_dir="./results/gpt",
            max_conv_turn=3,
            max_perspective=3,
            search_top_k=3,
            max_thread_num=3,
        )

        rm = YouRM(ydc_api_key=os.getenv("YDC_API_KEY"), k=engine_args.search_top_k)
        self.runner = STORMWikiRunner(engine_args, lm_configs, rm)

    def default(self, line):
        """Handle any input that's not a command as a query"""
        if line.startswith('/'):
            print(f"Unknown command: {line}")
            return
        self.generate_article(line)

    def do_help(self, arg):
        """Show help message"""
        self.print_help()

    def print_help(self):
        """Display available commands and usage"""
        help_text = f"""
{Fore.CYAN}STORM CLI Commands{Style.RESET_ALL}
{Fore.GREEN}/help{Style.RESET_ALL} - Show this help message
{Fore.GREEN}/exit{Style.RESET_ALL} - Exit the program
{Fore.GREEN}/open{Style.RESET_ALL} - Open last generated article in browser
{Fore.GREEN}/clear{Style.RESET_ALL} - Clear the screen
{Fore.GREEN}/version{Style.RESET_ALL} - Show STORM version

Enter a query to generate an article (e.g. \"AI in healthcare\")
"""
        print(help_text)

    def do_exit(self, arg):
        """Exit the STORM CLI"""
        print("\nThank you for using STORM!")
        return True

    def do_EOF(self, arg):
        """Exit on Ctrl-D"""
        print()  # Empty line for aesthetics
        return self.do_exit(arg)

    def emptyline(self):
        """Do nothing on empty line"""
        pass

    def do_open(self, arg):
        """Open last generated article in browser"""
        topic_dir = os.path.join("./results/gpt", self.runner.topic.replace(' ', '_'))
        md_path = os.path.join(topic_dir, "storm_gen_article.md")
        abs_md_path = os.path.abspath(md_path)
        webbrowser.open(f"file://{abs_md_path}")

    def do_clear(self, arg):
        """Clear the screen"""
        os.system('clear')

    def do_version(self, arg):
        """Show STORM version"""
        print(f"STORM CLI v0.1.0")

    def sanitize_filename(self, name):
        """Remove unsafe characters from filenames"""
        safe_name = re.sub(r'[\\/*?:"<>|()]', '', name)
        return safe_name.strip().replace(' ', '_')

    def generate_article(self, topic):
        """Generate an article about the given topic using default parameters"""
        try:
            # Sanitize topic name and create timestamp
            safe_topic = self.sanitize_filename(topic)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            topic_dir = os.path.join("./results/gpt", f"{safe_topic}_{timestamp}")
            
            os.makedirs(topic_dir, exist_ok=True)
            print(f"\nGenerating article about: {topic}")
            print("This may take several minutes...\n")
            
            self.runner.run(
                topic=topic,
                do_research=True,
                do_generate_outline=True,
                do_generate_article=True,
            )
            self.runner.post_run()
            self.runner.summary()
            
            # Create markdown version with metadata
            article_path = os.path.join(topic_dir, "storm_gen_article.txt")
            md_path = os.path.join(topic_dir, "storm_gen_article.md")
            
            with open(article_path, 'r') as f:
                article_content = f.read()
            
            # Create markdown version with metadata
            with open(md_path, 'w') as f:
                f.write(f"# {topic}\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Source: {os.path.abspath(article_path)}\n\n")
                f.write("---\n\n")
                f.write(article_content)
            
            # Print article to console
            print("\n" + "="*80 + "\n")
            print(article_content)
            print("\n" + "="*80 + "\n")
            
            # Show clickable link
            abs_md_path = os.path.abspath(md_path)
            print(f"\nMarkdown version: file://{abs_md_path}")
            
        except Exception as e:
            print(f"Error generating article: {str(e)}")

    def onecmd(self, line):
        if line.startswith('/'):
            if line == '/exit':
                return self.do_exit(line)
            elif line == '/help':
                self.print_help()
                return False
            elif line == '/open':
                self.do_open(line)
                return False
            elif line == '/clear':
                self.do_clear(line)
                return False
            elif line == '/version':
                self.do_version(line)
                return False
            else:
                print(f"{Fore.YELLOW}Unknown command: {line}{Style.RESET_ALL}")
                return False
        return super().onecmd(line)

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    if not os.getenv("YDC_API_KEY"):
        print("Error: YDC_API_KEY environment variable not set")
        sys.exit(1)

    StormCLI().cmdloop()

if __name__ == "__main__":
    main()
