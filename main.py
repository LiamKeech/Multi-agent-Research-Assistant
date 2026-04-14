from conversation_manager import ConversationManager

def main():
    manager = ConversationManager()
    print('=== Research Assistant (Conversational Mode) ===')
    print('Chat normally, or say "research <topic>" to trigger a report.')
    print('Type "quit" to exit.\n')

    while True:
        user_input = input('You: ').strip()
        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'q'):
            print('Goodbye!')
            break

        response = manager.send(user_input)
        print(f'\nAssistant: {response}\n')

if __name__ == '__main__': main()