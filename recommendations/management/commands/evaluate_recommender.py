from django.core.management.base import BaseCommand

from recommendations.services.recommender import train_test_split_evaluate


class Command(BaseCommand):
    help = 'Evaluate recommender with train/test split and report precision, recall, F1.'

    def add_arguments(self, parser):
        parser.add_argument('--test_ratio', type=float, default=0.2)
        parser.add_argument('--top_n', type=int, default=10)
        parser.add_argument('--max_users', type=int, default=100, help='Maximum number of users to evaluate (for performance)')

    def handle(self, *args, **options):
        self.stdout.write('Starting evaluation...')
        metrics = train_test_split_evaluate(
            test_ratio=options['test_ratio'], 
            top_n=options['top_n'],
            max_users=options['max_users']
        )
        self.stdout.write(self.style.SUCCESS(f"Precision: {metrics['precision']:.4f}"))
        self.stdout.write(self.style.SUCCESS(f"Recall:    {metrics['recall']:.4f}"))
        self.stdout.write(self.style.SUCCESS(f"F1:        {metrics['f1']:.4f}"))
        self.stdout.write(self.style.SUCCESS(f"Users evaluated: {metrics['users_evaluated']}"))


