from orchestrator import ResearchOrchestrator

def main():
    orchestrator = ResearchOrchestrator()

    print('=== Research Assistant ===')
    print('Type a research topic and press Enter.')
    print('Type "quit" to exit.\n')

    while True:
        query = input('Research topic: ').strip()
        if query.lower() in ('quit', 'exit', 'q'):
            print('Goodbye!')
            break
        if not query:
            continue

        report = orchestrator.run(query)
        print('\n' + '=' * 50)
        print(report)
        print('=' * 50 + '\n')

if __name__ == '__main__':
            main()