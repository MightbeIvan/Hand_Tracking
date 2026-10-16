
def main():
    while True:
        print("\n" + "=" * 36)
        print("       COMPUTER VISION")
        print("=" * 36)
        print("1. Hand Tracking")
        print("2. Eye Tracking")
        print("3. Exit")
        print("=" * 36)

        choice = input("Choose an option: ").strip()

        if choice == "1":
            import hand_tracking
            hand_tracking.main()

        elif choice == "2":
            import eye_tracking
            eye_tracking.main()

        elif choice == "3":
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please select 1, 2, or 3.")


if __name__ == "__main__":
    main()
