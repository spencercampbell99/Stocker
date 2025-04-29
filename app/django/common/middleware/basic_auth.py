import base64
from django.http import HttpResponse
from django.conf import settings

class BasicAuthMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        self.username = getattr(settings, 'BASIC_AUTH_USERNAME', 'admin')
        self.password = getattr(settings, 'BASIC_AUTH_PASSWORD', 'password')

    def __call__(self, request):
        if not request.META.get('HTTP_AUTHORIZATION'):
            return self._unauthorized()

        auth_type, credentials = request.META['HTTP_AUTHORIZATION'].split(' ', 1)

        if auth_type.lower() != 'basic':
            return self._unauthorized()

        decoded = base64.b64decode(credentials).decode('utf-8')
        username, password = decoded.split(':', 1)

        if username != self.username or password != self.password:
            return self._unauthorized()

        response = self.get_response(request)
        return response

    def _unauthorized(self):
        response = HttpResponse('Unauthorized', status=401)
        response['WWW-Authenticate'] = 'Basic realm="Restricted Area"'
        return response
