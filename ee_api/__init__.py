import ee


def is_authorized():
    try:
        ee.Initialize()
        print('Authorized')
    except Exception as e:
        print('You are not authorized: {}'.format(e))
        exit(1)
    return None


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
