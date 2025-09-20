from django.db import models
from django.contrib.auth.models import User


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    favorite_genres = models.JSONField(default=list, blank=True)
    preferred_ratings = models.JSONField(default=list, blank=True)  # Store user's movie ratings
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username}'s Profile"


class Movie(models.Model):
    movie_id = models.IntegerField(unique=True, db_index=True)
    title = models.CharField(max_length=255)
    year = models.IntegerField(null=True, blank=True)
    genres = models.CharField(max_length=255)

    def __str__(self):
        return f"{self.title} ({self.year})" if self.year else self.title


class Tag(models.Model):
    user_id = models.IntegerField(db_index=True)
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE, related_name='tags')
    tag = models.CharField(max_length=255)
    timestamp = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"{self.tag}"


class Rating(models.Model):
    user_id = models.IntegerField(db_index=True)
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE, related_name='ratings')
    rating = models.FloatField()
    timestamp = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = ('user_id', 'movie')

    def __str__(self):
        mid = self.movie.movie_id if self.movie_id else 'NA'
        return f"u{self.user_id}-m{mid}:{self.rating}"


class UserRating(models.Model):
    """Ratings from new users (separate from MovieLens data)"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='user_ratings')
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE, related_name='user_ratings')
    rating = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'movie')

    def __str__(self):
        return f"{self.user.username} rated {self.movie.title}: {self.rating}"
