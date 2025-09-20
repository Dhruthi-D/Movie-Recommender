from django.contrib import admin
from .models import UserProfile, Movie, Rating, Tag, UserRating


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "favorite_genres", "created_at")
    search_fields = ("user__username", "user__email")


@admin.register(Movie)
class MovieAdmin(admin.ModelAdmin):
    list_display = ("movie_id", "title", "year", "genres")
    search_fields = ("title", "genres")


@admin.register(Rating)
class RatingAdmin(admin.ModelAdmin):
    list_display = ("user_id", "movie", "rating")
    search_fields = ("user_id",)
    list_filter = ("rating",)


@admin.register(Tag)
class TagAdmin(admin.ModelAdmin):
    list_display = ("user_id", "movie", "tag")
    search_fields = ("tag",)

@admin.register(UserRating)
class UserRatingAdmin(admin.ModelAdmin):
    list_display = ("user", "movie", "rating", "created_at")
    list_filter = ("rating", "created_at")
    search_fields = ("user__username", "movie__title")
    ordering = ("-created_at",)

# Register your models here.
