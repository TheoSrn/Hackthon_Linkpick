FROM nginx:alpine

# Install curl for healthcheck
RUN apk add --no-cache curl

COPY index.html /usr/share/nginx/html/

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
