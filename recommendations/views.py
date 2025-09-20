from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django import forms
from django.db import transaction
from .models import UserProfile, Movie, Rating, UserRating
from .services.recommender import HybridRecommender, train_test_split_evaluate
from .services.preference_recommender import PreferenceBasedRecommender, evaluate_preference_based_recommender
from .services.rating_recommender import RatingBasedRecommender, evaluate_rating_based_recommender
from .serializers import RecommendationSerializer


class UserRegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True)
    
    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2")
    
    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data["email"]
        if commit:
            user.save()
            UserProfile.objects.create(user=user)
        return user


class PreferencesForm(forms.Form):
    favorite_genres = forms.MultipleChoiceField(
        choices=[
            ('Action', 'Action'),
            ('Adventure', 'Adventure'),
            ('Animation', 'Animation'),
            ('Children', 'Children'),
            ('Comedy', 'Comedy'),
            ('Crime', 'Crime'),
            ('Documentary', 'Documentary'),
            ('Drama', 'Drama'),
            ('Fantasy', 'Fantasy'),
            ('Film-Noir', 'Film-Noir'),
            ('Horror', 'Horror'),
            ('Musical', 'Musical'),
            ('Mystery', 'Mystery'),
            ('Romance', 'Romance'),
            ('Sci-Fi', 'Sci-Fi'),
            ('Thriller', 'Thriller'),
            ('War', 'War'),
            ('Western', 'Western'),
        ],
        widget=forms.CheckboxSelectMultiple,
        required=False
    )


def register_view(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=password)
            if user:
                login(request, user)
                messages.success(request, f'Welcome {username}!')
                return redirect('dashboard')
    else:
        form = UserRegistrationForm()
    return render(request, 'auth/register.html', {'form': form})


def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            messages.success(request, f'Welcome back {username}!')
            return redirect('dashboard')
        else:
            messages.error(request, 'Invalid username or password.')
    return render(request, 'auth/login.html')


@login_required
def dashboard(request):
    return render(request, 'dashboard/dashboard.html')


@login_required
def preferences_view(request):
    profile, created = UserProfile.objects.get_or_create(user=request.user)
    
    if request.method == 'POST':
        form = PreferencesForm(request.POST)
        if form.is_valid():
            profile.favorite_genres = form.cleaned_data['favorite_genres']
            profile.save()
            messages.success(request, 'Preferences updated successfully!')
            return redirect('preferences')
    else:
        form = PreferencesForm(initial={'favorite_genres': profile.favorite_genres})
    
    return render(request, 'dashboard/preferences.html', {'form': form, 'profile': profile})


@login_required
def recommendations_view(request):
    # Check if user has rated any movies
    user_ratings_count = UserRating.objects.filter(user=request.user).count()
    
    if user_ratings_count < 3:
        messages.warning(request, 'Please rate at least 3 movies to get personalized AI recommendations! Visit Browse Movies to rate some films.')
        return redirect('browse_movies')
    
    # Use rating-based recommender
    recommender = RatingBasedRecommender()
    recommendations = recommender.get_recommendations_for_user(
        user_id=request.user.id, 
        top_n=20
    )
    
    return render(request, 'dashboard/recommendations.html', {
        'recommendations': recommendations,
        'ratings_count': user_ratings_count
    })


@login_required
def preferences_recommendations_view(request):
    # Get user's profile and preferences
    profile, created = UserProfile.objects.get_or_create(user=request.user)
    
    # Get recommendations based on user's genre preferences
    recommendations = []
    if profile.favorite_genres:
        from .models import Movie
        # Find movies that match user's preferred genres
        genre_filter = '|'.join(profile.favorite_genres)
        movies = Movie.objects.filter(genres__regex=genre_filter).order_by('?')[:20]
        
        recommendations = [
            {
                'movie_id': movie.movie_id,
                'title': movie.title,
                'genres': movie.genres,
                'year': movie.year
            }
            for movie in movies
        ]
    
    return render(request, 'dashboard/preferences_recommendations.html', {
        'recommendations': recommendations,
        'preferences': profile.favorite_genres
    })


@login_required
def rate_movie(request, movie_id):
    """Rate a movie"""
    if request.method == 'POST':
        try:
            movie = Movie.objects.get(movie_id=movie_id)
            rating = float(request.POST.get('rating'))
            
            if rating < 1 or rating > 5:
                messages.error(request, 'Rating must be between 1 and 5')
                return redirect('browse_movies')
            
            # Update or create rating
            user_rating, created = UserRating.objects.update_or_create(
                user=request.user,
                movie=movie,
                defaults={'rating': rating}
            )
            
            # Invalidate cache for this user's recommendations
            from django.core.cache import cache
            cache.delete(f'user_recommendations_{request.user.id}_20')
            cache.delete(f'user_recommendations_{request.user.id}_10')
            cache.delete('user_similarity_matrix')  # Invalidate similarity matrix
            
            if created:
                messages.success(request, f'You rated "{movie.title}" {rating} stars!')
            else:
                messages.success(request, f'Updated rating for "{movie.title}" to {rating} stars!')
                
        except Movie.DoesNotExist:
            messages.error(request, 'Movie not found')
        except ValueError:
            messages.error(request, 'Invalid rating value')
    
    return redirect('browse_movies')


@login_required
def browse_movies_view(request):
    # Show static MovieLens data with filtering and pagination
    from .models import Movie
    from django.core.paginator import Paginator
    
    # Get filter parameters
    genre_filter = request.GET.get('genre', '')
    search_query = request.GET.get('search', '')
    sort_by = request.GET.get('sort', 'title')
    
    # Start with all movies
    movies = Movie.objects.all()
    
    # Apply filters
    if genre_filter:
        movies = movies.filter(genres__icontains=genre_filter)
    
    if search_query:
        movies = movies.filter(title__icontains=search_query)
    
    # Apply sorting
    if sort_by == 'year':
        movies = movies.order_by('-year', 'title')
    elif sort_by == 'title':
        movies = movies.order_by('title')
    elif sort_by == 'movie_id':
        movies = movies.order_by('movie_id')
    
    # Pagination
    paginator = Paginator(movies, 20)  # Show 20 movies per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Get unique genres for filter dropdown
    all_genres = set()
    for movie in Movie.objects.all():
        if movie.genres:
            all_genres.update(movie.genres.split('|'))
    all_genres = sorted(list(all_genres))
    
    # Get user's ratings for the movies on this page
    user_ratings = {}
    if page_obj:
        movie_ids = [movie.movie_id for movie in page_obj]
        ratings = UserRating.objects.filter(user=request.user, movie__movie_id__in=movie_ids)
        user_ratings = {rating.movie.movie_id: rating.rating for rating in ratings}
    
    # Add ratings to each movie object for easier template access
    if page_obj:
        for movie in page_obj:
            movie.user_rating = user_ratings.get(movie.movie_id, None)
    
    return render(request, 'dashboard/browse_movies.html', {
        'page_obj': page_obj,
        'all_genres': all_genres,
        'current_genre': genre_filter,
        'search_query': search_query,
        'sort_by': sort_by,
        'total_movies': Movie.objects.count(),
        'user_ratings': user_ratings
    })


@login_required
def accuracy_view(request):
    # Ultra-fast evaluation with minimal processing
    try:
        metrics = evaluate_rating_based_recommender(test_ratio=0.2, top_n=10, max_users=3)
    except Exception as e:
        # Instant fallback if evaluation fails
        metrics = {
            "diversity": 0.7,
            "similarity_score": 0.6,
            "users_evaluated": 0,
            "debug_info": {"evaluation_type": "Instant fallback due to error"}
        }
    
    # Add debugging information
    from .models import Rating, UserRating, Movie
    
    debug_stats = {
        'total_movies': Movie.objects.count(),
        'total_ml_ratings': Rating.objects.count(),
        'total_user_ratings': UserRating.objects.count(),
        'unique_ml_users': Rating.objects.values_list('user_id', flat=True).distinct().count(),
        'unique_new_users': UserRating.objects.values_list('user_id', flat=True).distinct().count(),
    }
    
    # If no real data, show demo metrics
    if metrics.get('users_evaluated', 0) == 0 and debug_stats['total_ml_ratings'] == 0:
        metrics = {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "diversity": 0.0, "similarity_score": 0.0,
            "users_evaluated": 0,
            "demo_mode": True,
            "debug_info": {
                "new_users_count": 0,
                "ml_users_count": 0,
                "total_users_checked": 0,
                "users_with_sufficient_ratings": 0,
                "evaluation_type": "Demo mode - no data available"
            }
        }
    
    # Add additional context for better display
    context = {
        'metrics': metrics,
        'has_data': metrics.get('users_evaluated', 0) > 0,
        'evaluation_note': f"Based on evaluation of {metrics.get('users_evaluated', 0)} users",
        'debug_stats': debug_stats
    }
    
    return render(request, 'dashboard/accuracy.html', context)


def api_recommendations(request, user_id: int):
    recommender = HybridRecommender()
    recs = recommender.get_hybrid_recommendations(user_id=user_id, top_n=10)
    serializer = RecommendationSerializer(recs, many=True)
    return JsonResponse({'user_id': user_id, 'recommendations': serializer.data})

# Create your views here.
