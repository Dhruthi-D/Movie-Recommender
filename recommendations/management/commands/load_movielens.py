import csv
import datetime as dt
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from recommendations.models import Movie, Rating, Tag


class Command(BaseCommand):
    help = "Load MovieLens dataset (ml-latest-small format) into the database. Provide --path to the unzipped folder."

    def add_arguments(self, parser):
        parser.add_argument('--path', type=str, required=True, help='Path to MovieLens folder (contains movies.csv, ratings.csv, tags.csv)')

    def handle(self, *args, **options):
        base = Path(options['path'])
        movies_csv = base / 'movies.csv'
        ratings_csv = base / 'ratings.csv'
        tags_csv = base / 'tags.csv'

        if not movies_csv.exists() or not ratings_csv.exists():
            raise CommandError('movies.csv and ratings.csv are required in the provided path')

        self.stdout.write('Loading movies...')
        self._load_movies(movies_csv)
        self.stdout.write('Loading ratings...')
        self._load_ratings(ratings_csv)
        if tags_csv.exists():
            self.stdout.write('Loading tags...')
            self._load_tags(tags_csv)
        self.stdout.write(self.style.SUCCESS('MovieLens data loaded successfully.'))

    @transaction.atomic
    def _load_movies(self, csv_path: Path):
        with csv_path.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            to_create = []
            for row in reader:
                movie_id = int(row['movieId'])
                title = row['title']
                year = None
                # Attempt to parse year from title e.g., Toy Story (1995)
                if title.endswith(')') and '(' in title:
                    try:
                        year = int(title.split('(')[-1].rstrip(')'))
                    except Exception:
                        year = None
                genres = row.get('genres', '')
                to_create.append(Movie(movie_id=movie_id, title=title, year=year, genres=genres))
            Movie.objects.bulk_create(to_create, ignore_conflicts=True, batch_size=1000)

    @transaction.atomic
    def _load_ratings(self, csv_path: Path):
        with csv_path.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            to_create = []
            for i, row in enumerate(reader, start=1):
                user_id = int(row['userId'])
                movie_id = int(row['movieId'])
                rating = float(row['rating'])
                ts = None
                if row.get('timestamp'):
                    try:
                        ts = dt.datetime.fromtimestamp(int(row['timestamp']))
                    except Exception:
                        ts = None
                try:
                    movie = Movie.objects.get(movie_id=movie_id)
                except Movie.DoesNotExist:
                    continue
                to_create.append(Rating(user_id=user_id, movie=movie, rating=rating, timestamp=ts))
                if len(to_create) >= 5000:
                    Rating.objects.bulk_create(to_create, ignore_conflicts=True, batch_size=5000)
                    to_create = []
            if to_create:
                Rating.objects.bulk_create(to_create, ignore_conflicts=True, batch_size=5000)

    @transaction.atomic
    def _load_tags(self, csv_path: Path):
        with csv_path.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            to_create = []
            for row in reader:
                user_id = int(row['userId'])
                movie_id = int(row['movieId'])
                tag = row.get('tag', '')
                ts = None
                if row.get('timestamp'):
                    try:
                        ts = dt.datetime.fromtimestamp(int(row['timestamp']))
                    except Exception:
                        ts = None
                try:
                    movie = Movie.objects.get(movie_id=movie_id)
                except Movie.DoesNotExist:
                    continue
                to_create.append(Tag(user_id=user_id, movie=movie, tag=tag, timestamp=ts))
                if len(to_create) >= 5000:
                    Tag.objects.bulk_create(to_create, ignore_conflicts=True, batch_size=5000)
                    to_create = []
            if to_create:
                Tag.objects.bulk_create(to_create, ignore_conflicts=True, batch_size=5000)


