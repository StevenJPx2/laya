import Foundation

/// Starter task for `laya-distill init --template routing`: route a support
/// message to billing, technical, or sales, with `other` as the fallback for
/// uncertain answers. Laya labels 60 unlabeled pool messages for training; 48
/// hand-labeled messages are evaluation-only gold. All messages are synthetic.
public enum RoutingTemplate {
    public static let spec = #"""
    {
      "spec_version": 2,
      "name": "routing",
      "version": "0.1.0",
      "description": "Route a customer support message to the team that should handle it.",
      "instructions": "A customer sent the support message below. Which team should handle it?",
      "input": {
        "fields": [
          {"name": "subject", "type": "string", "required": false, "max_chars": 200},
          {"name": "body", "type": "string", "max_chars": 4000}
        ]
      },
      "labels": [
        {"name": "billing", "description": "invoices, payments, charges, refunds, and existing subscriptions"},
        {"name": "technical", "description": "bugs, errors, outages, login problems, and how to use the product"},
        {"name": "sales", "description": "new purchases, quotes, demos, and pricing for prospective customers"},
        {"name": "other", "description": "anything else, or no clear team"}
      ],
      "abstain": {"label": "other", "min_confidence": 0.5},
      "teacher": {"question_type": "choice", "min_confidence": 0.6, "min_margin": 0.2, "uncertain": "abstain"},
      "dataset": {"max_examples": 5000, "holdout_fraction": 0.3, "split_seed": "routing-v1"},
      "student": {"hash_dimensions": 4096, "epochs": 300, "learning_rate": 0.05, "l2_grid": [0.0001, 0.001, 0.01, 0.1], "class_weighting": "balanced"}
    }
    """#

    /// Evaluation-only gold rows: (gold, subject, body).
    static let goldRows: [(String, String, String)] = [
        ("billing", "Charged twice", "My card was charged twice for the March invoice. Please refund one of them."),
        ("billing", "Refund request", "I cancelled last week but was still billed. I would like my money back."),
        ("billing", "Invoice copy", "Can you send me a PDF copy of invoice 2291 for our accounting team?"),
        ("billing", "Update card", "Our company card expired. How do I change the payment method on file?"),
        ("billing", "Wrong amount", "The invoice shows 40 seats but we only have 25 users. Please correct the charge."),
        ("billing", "VAT number", "Please add our VAT number to future invoices and reissue the last one."),
        ("billing", "Annual renewal", "We were auto-renewed for a year without notice. Can the renewal charge be reversed?"),
        ("billing", "Payment failed", "I got an email saying my payment failed, but my bank says the card is fine."),
        ("billing", "Downgrade billing", "We downgraded our plan mid-cycle. Will we get a prorated credit on the next bill?"),
        ("billing", "Receipt", "I need a receipt for the payment I made on the 3rd for an expense report."),
        ("billing", "Currency", "Can we be billed in euros instead of dollars from now on?"),
        ("billing", "Cancel subscription", "Please cancel my subscription at the end of this billing period."),
        ("technical", "Login broken", "I can't log in. The page says my password is wrong even after I reset it."),
        ("technical", "App crashes", "The desktop app crashes every time I open the settings screen."),
        ("technical", "API errors", "Our integration started getting 500 errors from your API this morning."),
        ("technical", "Sync not working", "Files I upload on my phone never appear on my laptop."),
        ("technical", "Export fails", "Exporting a report to CSV spins forever and never downloads."),
        ("technical", "Two-factor", "I lost my phone and can't get the two-factor code to sign in."),
        ("technical", "Slow dashboard", "The dashboard takes over a minute to load since yesterday's update."),
        ("technical", "Webhook setup", "How do I configure a webhook to notify our server when a task is completed?"),
        ("technical", "Error message", "I keep seeing 'session expired' every few minutes while working."),
        ("technical", "Import broken", "Importing contacts from a spreadsheet drops every row with an accent in the name."),
        ("technical", "Notifications", "Email notifications stopped arriving for everyone on our team."),
        ("technical", "Browser support", "The editor doesn't render correctly in Safari. Buttons overlap the text."),
        ("sales", "Pricing question", "We're evaluating tools for a 200-person team. What would enterprise pricing look like?"),
        ("sales", "Demo request", "Could someone give our team a product demo next week?"),
        ("sales", "Quote needed", "Please send a quote for 50 licenses so procurement can approve it."),
        ("sales", "Nonprofit discount", "Do you offer discounts for nonprofit organizations considering your product?"),
        ("sales", "Trial extension", "Our trial ends Friday and we're still deciding. Can we talk to someone about buying?"),
        ("sales", "Compare plans", "What's the difference between the Team and Business plans before we purchase?"),
        ("sales", "Security review", "Before we buy, our security team needs your SOC 2 report and a call with sales."),
        ("sales", "Reseller", "We're a consultancy interested in reselling your product to our clients."),
        ("sales", "Volume licensing", "We want to buy licenses for three subsidiaries. Is there volume pricing?"),
        ("sales", "Contract terms", "Can we get a two-year contract with a fixed price before signing up?"),
        ("sales", "New department", "Our marketing department wants to start using the product. Who can set up a purchase?"),
        ("sales", "Education pricing", "Is there special pricing for universities that want to roll this out to staff?"),
        ("other", "Job application", "I'd like to apply for the designer role listed on your careers page."),
        ("other", "Press inquiry", "I'm a journalist writing about remote work tools. Could I interview your CEO?"),
        ("other", "Partnership", "We run a podcast and would love to feature your founders in an episode."),
        ("other", "Thank you", "Just wanted to say your support team was great last week. No action needed."),
        ("other", "Office visit", "Do you have an office in Berlin we could visit?"),
        ("other", "Survey", "Would you fill out a short academic survey about software adoption?"),
        ("billing", "Double subscription", "It looks like I have two active subscriptions on two accounts. Please merge the billing."),
        ("technical", "Password reset email", "The password reset email never arrives, even in spam."),
        ("sales", "Upgrade inquiry", "We're a prospective customer comparing vendors. Can we see pricing for 500 seats?"),
        ("billing", "Tax exemption", "We're tax exempt. How do we stop sales tax from being added to our bills?"),
        ("technical", "Data missing", "Several projects disappeared from my workspace after I logged in today."),
        ("sales", "Pilot program", "We'd like to run a paid pilot with one team before a company-wide purchase."),
    ]

    /// Unlabeled training pool for Laya to label: (subject, body). Distinct from every gold row.
    static let poolRows: [(String, String)] = [
        ("Overcharged", "I was billed for the premium tier but I'm on the basic plan."),
        ("Refund", "Please refund the charge from yesterday, I signed up by mistake."),
        ("Billing address", "How do I change the billing address that appears on our invoices?"),
        ("Invoice question", "Why does this month's invoice have an extra line item for storage?"),
        ("Card declined", "My card keeps getting declined when I try to pay the outstanding balance."),
        ("Credit note", "We were promised a credit for the outage. When will it show on our invoice?"),
        ("Pay by invoice", "Can we pay by bank transfer instead of credit card for our existing account?"),
        ("Duplicate charge", "There are two identical charges on my statement from your company."),
        ("Billing contact", "Please send future invoices to finance@example.com instead of me."),
        ("Subscription end", "When does my current subscription period end? I don't want to be charged again."),
        ("Prorated refund", "I removed ten seats. Will I get a partial refund for the unused time?"),
        ("Late fee", "Why was a late fee added when I paid on the due date?"),
        ("Coupon", "My discount code wasn't applied to the last charge. Can you fix the bill?"),
        ("Payment receipt", "I can't find the receipt for last quarter's payment in the dashboard."),
        ("Invoice due date", "Can the invoice due date be moved to the end of the month?"),
        ("Error on save", "Every time I click save I get an 'unexpected error' banner and my changes are lost."),
        ("Can't upload", "Uploading images larger than 5 MB fails with a network error."),
        ("Account locked", "My account got locked after too many login attempts. How do I unlock it?"),
        ("Integration issue", "The Slack integration stopped posting messages to our channel."),
        ("Mobile bug", "On Android the app freezes when I scroll through long lists."),
        ("Search broken", "Search returns no results even for documents I know exist."),
        ("SSO setup", "How do I set up single sign-on with our identity provider?"),
        ("Outage?", "Is the service down? Nothing loads for anyone in our office."),
        ("Timezone bug", "All my calendar events show up one hour off after the daylight saving change."),
        ("API rate limit", "We're hitting rate limit errors far below the documented limit."),
        ("Printing", "Printed reports cut off the last column of every table."),
        ("Permissions", "How do I give a teammate edit access to only one project?"),
        ("Dark mode bug", "In dark mode some text is black on a dark background and unreadable."),
        ("Keyboard shortcuts", "The keyboard shortcuts stopped working after the latest update."),
        ("Backup restore", "How do I restore a project from yesterday's backup?"),
        ("Enterprise plan", "We're a bank looking at your enterprise plan. Can we schedule a call?"),
        ("Pricing for startups", "Do you have startup pricing? We're a new company evaluating options."),
        ("Buy more seats", "We're not customers yet, but we'd like pricing for 30 seats."),
        ("Demo", "I'd love a walkthrough of the product for my leadership team."),
        ("Procurement", "Our procurement team needs a formal quote and your standard contract."),
        ("Competitor switch", "We're considering switching from a competitor. What would it cost for our team?"),
        ("Government pricing", "Do you offer pricing for government agencies interested in purchasing?"),
        ("Free trial", "Can we start a free trial for our whole department before buying?"),
        ("Partner pricing", "As a prospective agency partner, what margins do you offer on resale?"),
        ("Sales call", "Please have a salesperson contact me about licensing for my company."),
        ("Multi-year deal", "Would you discount a three-year upfront purchase for a new customer?"),
        ("Pricing page", "Your pricing page doesn't list the Enterprise tier. Who can quote it for us?"),
        ("Evaluation", "We're comparing three vendors this month. Could we get a proposal?"),
        ("Pilot", "Could we buy a small number of licenses first to evaluate before a larger order?"),
        ("Rollout", "We want to purchase for 1,000 employees next quarter. Who handles that?"),
        ("Internship", "Are you offering summer internships for software engineering students?"),
        ("Event invite", "We'd like to invite your team to speak at our conference in May."),
        ("Feedback", "I have some general thoughts about your company's mission, just sharing."),
        ("Merch", "Do you sell branded t-shirts or stickers?"),
        ("Charity", "Would your company sponsor our local charity run?"),
        ("Research", "I'm a student researching SaaS companies. Can I ask a few questions?"),
        ("Hello", "Hi, just testing whether this form works."),
        ("Refund and bug", "The app crashed during checkout and I was charged anyway. Please refund me."),
        ("Upgrade billing", "We upgraded to Business yesterday but the invoice still shows the old plan price."),
        ("Login and invoice", "I can't log in to download my invoice. The page keeps reloading."),
        ("Account deletion", "Please delete my account and all of my data."),
        ("Data export", "How can I export all of my data before our contract ends?"),
        ("Feature request", "It would be great if you supported Markdown tables in notes."),
        ("Payment method", "Do you accept PayPal for monthly subscription payments?"),
        ("Slow sync", "Changes take several minutes to sync between my devices."),
    ]

    public static var dataset: String {
        let gold = goldRows.enumerated().map { index, row in
            TemplateLine.encode(id: String(format: "gold-%03d", index + 1), group: nil, gold: row.0, input: ["subject": row.1, "body": row.2])
        }
        let pool = poolRows.enumerated().map { index, row in
            TemplateLine.encode(id: String(format: "pool-%03d", index + 1), group: nil, gold: nil, input: ["subject": row.0, "body": row.1])
        }

        return (pool + gold).joined(separator: "\n") + "\n"
    }
}

enum TemplateLine {
    static func encode(id: String, group: String?, gold: String?, input: [String: String]) -> String {
        var object: [String: Any] = ["id": id, "input": input]
        if let group { object["group"] = group }
        if let gold { object["gold"] = gold }

        let data = try! JSONSerialization.data(withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes])

        return String(decoding: data, as: UTF8.self)
    }
}

/// Built-in templates for `laya-distill init`.
public enum Templates {
    public static let names = ["routing", "permission"]

    public static func named(_ name: String) -> (spec: String, dataset: String)? {
        switch name {
        case "routing": return (RoutingTemplate.spec, RoutingTemplate.dataset)
        case "permission": return (PermissionTemplate.spec, PermissionTemplate.dataset)
        default: return nil
        }
    }
}
