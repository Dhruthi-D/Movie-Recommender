from django import template

register = template.Library()

@register.filter
def stars_display(rating):
    """Convert rating to star display HTML"""
    if not rating:
        return '<i class="far fa-star"></i>' * 5
    
    full_stars = int(rating)
    has_half_star = rating - full_stars >= 0.5
    empty_stars = 5 - full_stars - (1 if has_half_star else 0)
    
    stars_html = ''
    
    # Full stars
    stars_html += '<i class="fas fa-star"></i>' * full_stars
    
    # Half star if needed
    if has_half_star:
        stars_html += '<i class="fas fa-star-half-alt"></i>'
    
    # Empty stars
    stars_html += '<i class="far fa-star"></i>' * empty_stars
    
    return stars_html
