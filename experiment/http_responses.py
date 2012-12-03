import httplib

def status(code, name):
    msg = httplib.responses[code]
    if msg != name:
        raise ValueError("invalid code and/or name: %d %s" % (code, name))
    response = "Status: %d %s\n\n" % (code, name)
    return response

def content_type(mime):
    response = "Content-Type: %s\n\n" % mime
    return response
