# BUG-07 FIX: This file is intentionally empty.
#
# Menu CRUD operations are handled entirely by:
#     app/application/services/menu_service.py  (business logic)
#     app/presentation/api/v1/menu_routes.py    (HTTP layer)
#
# A separate "ManageMenuUseCase" class is not needed because the menu
# service is simple enough to be called directly from the routes via
# dependency injection. This stub is kept to maintain the use_cases/
# package structure for future sprint expansion.
