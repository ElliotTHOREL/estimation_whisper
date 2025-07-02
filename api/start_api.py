#!/usr/bin/env python3  
""")
    
    try:
        uvicorn.run(
            "speech_api:app",
            host=args.host,
            port=port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,  # reload mode nécessite 1 worker
            log_level=args.log_level,
            access_log=True
        )
    except Exception as e:
        print(f"Erreur lors du démarrage: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 